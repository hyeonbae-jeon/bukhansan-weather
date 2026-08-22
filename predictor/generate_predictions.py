"""
기상청 예보(격자 단위) + 학습된 모델(model_final)을 결합해서, 지점별 온도/습도 예측을
만들어 프론트엔드(카카오맵)가 바로 읽을 수 있는 JSON으로 저장한다.

사용법:
    python generate_predictions.py \
        --forecast ../data/weather/forecast.csv \
        --model-dir ../data/model_final \
        --sensor ../data/sensor/sensor_merged.csv \
        --out ../frontend/points_predictions.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from kma_grid import latlon_to_grid

# 기상청 단기예보의 하늘상태(SKY) 코드(1=맑음,3=구름많음,4=흐림)를
# 학습에 쓴 ASOS 전운량(0~10 정수) 스케일에 대략 맞춰 변환. 정밀 매핑이 아니라 근사치.
SKY_TO_CLOUD = {1: 1, 3: 6, 4: 9}


def apply_offset_single(point_id: str, hour: int, offset_map: dict, label: str) -> float:
    tod = "오전" if hour < 12 else "오후"
    key = f"{point_id}|{tod}"
    return offset_map.get(label, {}).get(key, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--current-obs", required=False, default=None,
                     help="fetch_current_obs.py가 만든 current_obs.json 경로 (선택)")
    args = ap.parse_args()

    current_obs = {}
    if args.current_obs and os.path.exists(args.current_obs):
        with open(args.current_obs, encoding="utf-8") as f:
            current_obs = json.load(f)
        print(f"실황 데이터 로드됨: {len(current_obs)}개 격자")
    else:
        print("실황 데이터 없음 (--current-obs 미지정 또는 파일 없음) — obs 필드는 null로 채워짐")

    forecast = pd.read_csv(args.forecast)
    forecast["fcstDateTime"] = pd.to_datetime(forecast["fcstDateTime"])
    for c in ["TMP", "REH", "WSD", "VEC", "SKY", "PTY", "POP"]:
        if c in forecast.columns:
            forecast[c] = pd.to_numeric(forecast[c], errors="coerce")

    with open(os.path.join(args.model_dir, "offsets.json"), encoding="utf-8") as f:
        offsets = json.load(f)
    with open(os.path.join(args.model_dir, "station_meta.json"), encoding="utf-8") as f:
        stations = pd.DataFrame(json.load(f))

    model_temp = lgb.Booster(model_file=os.path.join(args.model_dir, "model_final_temp.txt"))
    model_hum = lgb.Booster(model_file=os.path.join(args.model_dir, "model_final_humidity.txt"))

    stations["grid"] = stations.apply(
        lambda r: "{}_{}".format(*latlon_to_grid(r["GNSS-위도"], r["GNSS-경도"])), axis=1
    )

    results = []
    for _, st in stations.iterrows():
        fc = forecast[forecast["grid"] == st["grid"]].sort_values("fcstDateTime")
        if fc.empty:
            continue

        rows = []
        for _, f in fc.iterrows():
            hour = f["fcstDateTime"].hour
            month = f["fcstDateTime"].month
            dow = f["fcstDateTime"].dayofweek
            sky = f.get("SKY", np.nan)
            cloud = SKY_TO_CLOUD.get(int(sky), 5) if pd.notna(sky) else 5
            pty = f.get("PTY", 0)
            rain_proxy = 0.0 if pd.isna(pty) or pty == 0 else max(float(f.get("POP", 30)) / 100.0 * 3.0, 0.5)
            wind_rad = np.deg2rad(f.get("VEC", 0) if pd.notna(f.get("VEC", 0)) else 0)

            feat = pd.DataFrame([{
                "기온": f.get("TMP", np.nan),
                "기준습도": f.get("REH", np.nan),
                "강수량": rain_proxy,
                "풍속": f.get("WSD", np.nan),
                "전운량": cloud,
                "시간대": hour,
                "월": month,
                "요일": dow,
                "GNSS-경도": st["GNSS-경도"],
                "GNSS-위도": st["GNSS-위도"],
                "풍향_sin": np.sin(wind_rad),
                "풍향_cos": np.cos(wind_rad),
                "국가지점번호": st["국가지점번호"],
            }])
            feat["국가지점번호"] = feat["국가지점번호"].astype("category")

            offset_t = apply_offset_single(st["국가지점번호"], hour, offsets, "temp")
            offset_h = apply_offset_single(st["국가지점번호"], hour, offsets, "humidity")

            resid_t = model_temp.predict(feat)[0]
            resid_h = model_hum.predict(feat)[0]

            pred_temp = round(float(f.get("TMP", np.nan)) + offset_t + resid_t, 1) if pd.notna(f.get("TMP")) else None
            pred_hum = round(float(f.get("REH", np.nan)) + offset_h + resid_h, 1) if pd.notna(f.get("REH")) else None

            rows.append({
                "time": f["fcstDateTime"].strftime("%Y-%m-%dT%H:%M"),
                "temp": pred_temp,
                "humidity": pred_hum,
                # 아래는 모델 보정 없이 기상청 예보 원본값 그대로 (강수형태/하늘상태/강수확률/풍속)
                # 산악지형 위험요소(비/눈/소나기 등) 표시에 사용
                "pty": int(pty) if pd.notna(pty) else 0,
                "sky": int(sky) if pd.notna(sky) else None,
                "pop": float(f.get("POP")) if pd.notna(f.get("POP")) else None,
                "wind": float(f.get("WSD")) if pd.notna(f.get("WSD")) else None,
                # 진단용: 이 값이 어떻게 계산됐는지 투명하게 보여주기 위한 분해값
                # 최종예측 = 기상청예보원본값 + modelOffset(지점×시간대 평균편차) + modelResidual(그날 기상조건별 추가보정)
                "refTemp": round(float(f.get("TMP")), 1) if pd.notna(f.get("TMP")) else None,
                "refHumidity": round(float(f.get("REH")), 1) if pd.notna(f.get("REH")) else None,
                "modelOffsetTemp": round(float(offset_t), 2),
                "modelOffsetHumidity": round(float(offset_h), 2),
                "modelResidualTemp": round(float(resid_t), 2),
                "modelResidualHumidity": round(float(resid_h), 2),
            })

        # 지금 실제 관측값(있으면) — 예보가 아니라 진짜 지금 관측된 값
        obs_entry = current_obs.get(st["grid"]) if current_obs else None
        obs_out = None
        if obs_entry:
            obs_pty = obs_entry.get("PTY")
            obs_out = {
                "baseDateTime": obs_entry.get("baseDateTime"),
                "temp": obs_entry.get("T1H"),
                "humidity": obs_entry.get("REH"),
                "rain1h": obs_entry.get("RN1"),
                "pty": int(obs_pty) if obs_pty is not None else 0,
                "wind": obs_entry.get("WSD"),
            }

        results.append({
            "id": st["국가지점번호"],
            "name": st["다목적위치표지판번호"],
            "lat": st["GNSS-위도"],
            "lon": st["GNSS-경도"],
            "obs": obs_out,
            "forecasts": rows,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {args.out} ({len(results)}개 지점)")


if __name__ == "__main__":
    main()
