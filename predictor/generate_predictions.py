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
import re

import numpy as np
import pandas as pd
import lightgbm as lgb

from kma_grid import latlon_to_grid

# 기상청 단기예보의 하늘상태(SKY) 코드(1=맑음,3=구름많음,4=흐림)를
# 학습에 쓴 ASOS 전운량(0~10 정수) 스케일에 대략 맞춰 변환. 정밀 매핑이 아니라 근사치.
SKY_TO_CLOUD = {1: 1, 3: 6, 4: 9}


def parse_precip_amount(raw, unit):
    """기상청 PCP(강수량)/SNO(적설량) 원본값은 숫자가 아니라 "3.0mm", "1mm 미만",
    "강수없음"/"적설없음" 같은 문자열(가끔 범위 표기 "30.0~50.0mm"도 있음)이라
    그대로는 못 쓴다. 최대한 숫자(mm 또는 cm)로 바꾸고, 원본 문자열도 같이 돌려줘서
    프론트에서 "1mm 미만"처럼 애매한 경우엔 원문 그대로 보여줄 수 있게 한다.
    반환: (근사값 또는 None, 원본 문자열 또는 None)"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    s = str(raw).strip()
    if s in ("강수없음", "적설없음", "0", "0.0", ""):
        return 0.0, s
    if "미만" in s:
        m = re.search(r"([\d.]+)", s)
        return (round(float(m.group(1)) / 2, 1) if m else 0.5), s
    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return None, s
    vals = [float(n) for n in nums]
    return round(sum(vals) / len(vals), 1), s


def lookup_grid_weather(grid, t, forecast, ultra_forecast):
    """주어진 (격자, 시각)의 강수형태/하늘상태/강수확률/강수량을 돌려준다. 메인 예측
    루프와 같은 규칙(초단기예보 있으면 우선, 강수량은 PTY로 비/눈 구분)을 쓰는 걸
    build_interpolated_points()에서도 그대로 쓸 수 있게 뽑아둔 공용 함수."""
    frow = None
    if forecast is not None:
        sub = forecast[(forecast["grid"] == grid) & (forecast["fcstDateTime"] == t)]
        if not sub.empty:
            frow = sub.iloc[0]

    u = None
    if ultra_forecast is not None:
        sub = ultra_forecast[(ultra_forecast["grid"] == grid) & (ultra_forecast["fcstDateTime"] == t)]
        if not sub.empty:
            u = sub.iloc[0]

    def pick(ultra_col, regular_col, default=np.nan):
        if u is not None and pd.notna(u.get(ultra_col, np.nan)):
            return u[ultra_col]
        if frow is not None:
            return frow.get(regular_col, default)
        return default

    sky = pick("SKY", "SKY")
    pty = pick("PTY", "PTY", 0)
    pty_int = int(pty) if pd.notna(pty) else 0
    is_snow_type = pty_int in (3, 7)

    if not is_snow_type and u is not None and pd.notna(u.get("RN1", np.nan)):
        precip_raw = u.get("RN1")
    elif frow is not None:
        precip_raw = frow.get("SNO") if is_snow_type else frow.get("PCP")
    else:
        precip_raw = None
    precip_mm, precip_label = parse_precip_amount(precip_raw, "cm" if is_snow_type else "mm")

    return {
        "pty": pty_int,
        "sky": int(sky) if pd.notna(sky) else None,
        "pop": float(frow.get("POP")) if frow is not None and pd.notna(frow.get("POP")) else None,
        "precipMm": precip_mm,
        "precipUnit": "cm" if is_snow_type else "mm",
        "precipLabel": precip_label,
    }


# 센서 지점명 중 "북한00-00" 코드 형식이 아닌 예외들. 원래는 "족두리봉"에 가장 가까운
# 코드(북한65-02/북한65-03)를 붙이려 했는데, 둘 다 이미 다른 실측 센서가 쓰고 있는
# 코드라 중복이 생겨서(더 혼란스러움) 억지로 매핑하지 않기로 함 — 그냥 원래 이름 유지.
CODE_ALIASES = {}

# 표출에서 완전히 제외할 국가지점번호. 사유를 값으로 남겨둠.
EXCLUDED_STATION_IDS = {
    # "족두리봉"은 다른 지점들과 달리 "북한00-00"/"둘레길000-00" 코드 체계에 속하지
    # 않아서(장소명 그대로) 화면에 표시하지 않기로 함
    "다사50905778": "족두리봉 - 코드 체계에 맞지 않아 제외",
    # "둘레길105-02" 코드가 서로 다른 좌표의 센서 2개(다사54515721, 다사55305741)에
    # 중복 부여되어 있어 혼란스러움 - 데이터가 조금 더 최신/많은 다사55305741만 남기고 제외
    "다사54515721": "둘레길105-02 코드 중복 - 다사55305741만 남기고 제외",
}


def load_point_names(path):
    """전체 187개 지점의 '코드 -> 세부 지명' 매핑. 마커엔 코드만, 상세정보엔
    코드+세부지명을 같이 보여주기 위해 씀."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["id"], df["name"]))


def apply_offset_single(point_id: str, hour: int, offset_map: dict, label: str) -> float:
    tod = "오전" if hour < 12 else "오후"
    key = f"{point_id}|{tod}"
    return offset_map.get(label, {}).get(key, 0.0)


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def build_interpolated_points(extra_points_csv, real_results, forecast=None, ultra_forecast=None):
    """온습도 센서가 없는 지점들을, 실측 기반 모델 예측이 있는 주변 지점들로부터
    역거리가중(IDW)으로 보간해서 만든다. 모델 예측이 아니라 '추정치'임을
    interpolated:true로 명시해서 프론트에서 구분 표시할 수 있게 한다.

    강수형태/하늘상태/강수확률/강수량은 (예전엔 IDW 루프에서 "가장 먼저 만난 실측
    지점" 값을 그냥 갖다 썼는데, 그게 거리와 무관하게 리스트 순서상 우연히 먼저
    나온 지점이라 엉뚱한(먼) 지점의 값을 쓰는 경우가 있었다) 이제 이 지점 자신의
    좌표로 계산한 진짜 자기 격자에서 직접 가져온다 — 기온/습도만 IDW로 보간하고
    강수 관련 값은 실측 지점과 무관하게 그 위치의 실제 격자 예보를 그대로 씀."""
    if not extra_points_csv or not os.path.exists(extra_points_csv):
        return []

    extra_df = pd.read_csv(extra_points_csv)
    if extra_df.empty:
        return []

    # 모든 지점이 공유하는 예보 시각 목록 (첫 실측 지점 기준)
    times = [f["time"] for f in real_results[0]["forecasts"]] if real_results else []

    results = []
    for _, ep in extra_df.iterrows():
        ep_grid = "{}_{}".format(*latlon_to_grid(ep["lat"], ep["lon"]))

        forecasts = []
        for t in times:
            num_t, den_t, num_h, den_h = 0.0, 0.0, 0.0, 0.0
            for rp in real_results:
                f = next((x for x in rp["forecasts"] if x["time"] == t), None)
                if not f or f["temp"] is None:
                    continue
                d = haversine_km(ep["lat"], ep["lon"], rp["lat"], rp["lon"])
                if d < 0.01:
                    num_t, den_t = f["temp"], 1
                    num_h, den_h = (f["humidity"] or 0), 1
                    break
                w = 1 / (d ** 2)
                num_t += w * f["temp"]
                den_t += w
                if f["humidity"] is not None:
                    num_h += w * f["humidity"]
                    den_h += w

            temp = round(num_t / den_t, 1) if den_t > 0 else None
            hum = round(num_h / den_h, 1) if den_h > 0 else None

            weather = lookup_grid_weather(ep_grid, pd.Timestamp(t), forecast, ultra_forecast)
            forecasts.append({
                "time": t,
                "temp": temp,
                "humidity": hum,
                **weather,
                "wind": None,
            })

        results.append({
            "id": ep["id"],
            "code": ep["id"],
            "name": ep["name"],
            "detailName": ep["name"],
            "lat": ep["lat"],
            "lon": ep["lon"],
            "elevation_m": ep.get("elevation_m"),
            "obs": None,
            "interpolated": True,  # AI 모델 예측이 아니라 주변 지점 보간 추정치임을 표시
            "forecasts": forecasts,
        })

    return results


# "기상청 관측값" 패널에서 고를 수 있는 주변 행정동 — 프론트엔드(index.html)의
# REFERENCE_AREAS와 반드시 이름/좌표가 같아야 함(둘 다 수정할 것). 대부분 센서
# 지점의 nx,ny 격자(9개 중 4개뿐)에 이미 다 들어와서, 격자를 새로 더 부르지
# 않고도 이 지역들 각각의 진짜 자기 격자로 정확하게 매칭할 수 있다.
REFERENCE_AREAS = [
    {"name": "성북구 정릉동", "lat": 37.6068, "lon": 127.0089},
    {"name": "강북구 수유동", "lat": 37.6392, "lon": 127.0165},
    {"name": "강북구 우이동", "lat": 37.6633, "lon": 127.0122},
    {"name": "종로구 구기동", "lat": 37.6058, "lon": 126.9611},
    {"name": "종로구 평창동", "lat": 37.6114, "lon": 126.9706},
    {"name": "은평구 진관동", "lat": 37.6386, "lon": 126.9317},
    {"name": "은평구 불광동", "lat": 37.6106, "lon": 126.9296},
    {"name": "고양시 덕양구 효자동", "lat": 37.6584, "lon": 126.9615},
    {"name": "고양시 덕양구 북한동", "lat": 37.6693, "lon": 126.9515},
]


def build_reference_areas(current_obs: dict) -> list:
    """각 지역의 실제 좌표로 계산한 자기 격자(nx,ny)에서 직접 관측값을 가져온다.
    이전엔 "가장 가까운 센서 지점"을 거쳐서 관측값을 가져왔는데, 그 센서가 하필
    다른 격자에 속해 있으면 엉뚱한(하지만 인접한) 격자의 값을 보여주는 셈이라
    부정확할 수 있었음 — 이제 지역 좌표 → 격자를 직접 계산해서 그 문제를 없앤다."""
    out = []
    for area in REFERENCE_AREAS:
        nx, ny = latlon_to_grid(area["lat"], area["lon"])
        grid_key = f"{nx}_{ny}"
        obs_entry = current_obs.get(grid_key) if current_obs else None
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
        out.append({
            "name": area["name"],
            "lat": area["lat"],
            "lon": area["lon"],
            "grid": grid_key,
            "obs": obs_out,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-points", required=False, default=None,
                     help="온습도 센서가 없는 추가 지점 목록 CSV (id,name,lat,lon,elevation_m) - 선택")
    ap.add_argument("--current-obs", required=False, default=None,
                     help="fetch_current_obs.py가 만든 current_obs.json 경로 (선택)")
    ap.add_argument("--point-names", required=False, default=None,
                     help="지점 코드->세부지명 매핑 CSV (id,name) - 선택, data/point_names.csv")
    ap.add_argument("--reference-areas-out", required=False, default=None,
                     help="'기상청 관측값' 지역 선택용 별도 JSON 저장 경로 (선택, 예: ../frontend/reference_areas.json)")
    ap.add_argument("--ultra-forecast", required=False, default=None,
                     help="fetch_ultra_forecast.py가 만든 6시간 이내 초단기예보 CSV 경로 (선택) - "
                          "있으면 겹치는 시각의 기온/습도/강수형태/하늘상태/바람을 이걸로 우선 대체")
    args = ap.parse_args()

    point_names_map = load_point_names(args.point_names)
    if point_names_map:
        print(f"지점 세부지명 {len(point_names_map)}개 로드됨")

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

    ultra_forecast = None
    if args.ultra_forecast and os.path.exists(args.ultra_forecast):
        ultra_forecast = pd.read_csv(args.ultra_forecast)
        if not ultra_forecast.empty:
            ultra_forecast["fcstDateTime"] = pd.to_datetime(ultra_forecast["fcstDateTime"])
            for c in ["T1H", "REH", "WSD", "VEC", "SKY", "PTY"]:
                if c in ultra_forecast.columns:
                    ultra_forecast[c] = pd.to_numeric(ultra_forecast[c], errors="coerce")
            print(f"초단기예보 로드됨: {len(ultra_forecast)}행 (6시간 이내 시각은 이 값을 우선 사용)")
        else:
            ultra_forecast = None
    else:
        print("초단기예보 없음 (--ultra-forecast 미지정 또는 파일 없음) — 단기예보만 사용")

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
        if st["국가지점번호"] in EXCLUDED_STATION_IDS:
            continue
        fc = forecast[forecast["grid"] == st["grid"]].sort_values("fcstDateTime")
        if fc.empty:
            continue
        # 초단기예보(있으면)는 지금부터 6시간 이내만 커버하지만 매시 갱신되고
        # 소나기처럼 3시간 단위 단기예보가 놓치기 쉬운 급변 기상을 더 잘 잡는다 —
        # 시각이 겹치면 이쪽 값을 우선 씀. 격자별로 시각→행 조회가 빠르게 되도록 인덱싱.
        ultra_fc = None
        if ultra_forecast is not None:
            g = ultra_forecast[ultra_forecast["grid"] == st["grid"]]
            if not g.empty:
                ultra_fc = g.set_index("fcstDateTime")

        rows = []
        for _, f in fc.iterrows():
            hour = f["fcstDateTime"].hour
            month = f["fcstDateTime"].month
            dow = f["fcstDateTime"].dayofweek

            # 이 시각에 초단기예보 값이 있으면(=6시간 이내) 기온/습도/하늘상태/강수형태/
            # 바람을 그걸로 대체. 강수확률(POP)은 초단기예보에 없는 항목이라 항상
            # 단기예보(f) 값을 그대로 씀.
            u = None
            if ultra_fc is not None and f["fcstDateTime"] in ultra_fc.index:
                u = ultra_fc.loc[f["fcstDateTime"]]
                if isinstance(u, pd.DataFrame):  # 혹시 같은 시각이 중복으로 있으면 첫 행만
                    u = u.iloc[0]

            def pick(ultra_col, regular_col, regular_default=np.nan):
                if u is not None and pd.notna(u.get(ultra_col, np.nan)):
                    return u[ultra_col]
                return f.get(regular_col, regular_default)

            tmp_val = pick("T1H", "TMP")
            reh_val = pick("REH", "REH")
            sky = pick("SKY", "SKY")
            pty = pick("PTY", "PTY", 0)
            wsd_val = pick("WSD", "WSD")
            vec_val = pick("VEC", "VEC", 0)

            cloud = SKY_TO_CLOUD.get(int(sky), 5) if pd.notna(sky) else 5
            rain_proxy = 0.0 if pd.isna(pty) or pty == 0 else max(float(f.get("POP", 30)) / 100.0 * 3.0, 0.5)
            wind_rad = np.deg2rad(vec_val if pd.notna(vec_val) else 0)

            feat = pd.DataFrame([{
                "기온": tmp_val,
                "기준습도": reh_val,
                "강수량": rain_proxy,
                "풍속": wsd_val,
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

            pred_temp = round(float(tmp_val) + offset_t + resid_t, 1) if pd.notna(tmp_val) else None
            pred_hum = round(float(reh_val) + offset_h + resid_h, 1) if pd.notna(reh_val) else None

            # 강수량(비)/적설량(눈)은 지점별로 보정할 근거가 없다(센서가 온습도만 재고
            # 강수는 안 재서 학습 데이터가 없음) — 기상청 격자 예보값을 그대로 씀.
            # PTY로 비/눈 중 어느 쪽 수치를 봐야 하는지 정해서 그 필드만 파싱.
            pty_int = int(pty) if pd.notna(pty) else 0
            is_snow_type = pty_int in (3, 7)  # 3=눈, 7=눈날림
            # 초단기예보엔 적설량(SNO) 항목이 없어서, 눈 예보일 땐 그대로 단기예보의
            # SNO를 쓰고, 비 계열일 때만(RN1 있으면) 초단기예보 값을 우선한다.
            if not is_snow_type and u is not None and pd.notna(u.get("RN1", np.nan)):
                precip_raw = u.get("RN1")
            else:
                precip_raw = f.get("SNO") if is_snow_type else f.get("PCP")
            precip_mm, precip_label = parse_precip_amount(precip_raw, "cm" if is_snow_type else "mm")

            rows.append({
                "time": f["fcstDateTime"].strftime("%Y-%m-%dT%H:%M"),
                "temp": pred_temp,
                "humidity": pred_hum,
                # 아래는 모델 보정 없이 기상청 예보 원본값 그대로 (강수형태/하늘상태/강수확률/풍속)
                # 산악지형 위험요소(비/눈/소나기 등) 표시에 사용
                "pty": pty_int,
                "sky": int(sky) if pd.notna(sky) else None,
                "pop": float(f.get("POP")) if pd.notna(f.get("POP")) else None,
                "precipMm": precip_mm,        # 눈이면 cm 단위(적설량), 비면 mm(강수량) — precipUnit 참고
                "precipUnit": "cm" if is_snow_type else "mm",
                "precipLabel": precip_label,  # "1mm 미만"처럼 정확한 수치가 아닐 때 원문 그대로
                "wind": float(wsd_val) if pd.notna(wsd_val) else None,
                # 진단용: 이 값이 어떻게 계산됐는지 투명하게 보여주기 위한 분해값
                # 최종예측 = 기상청예보원본값 + modelOffset(지점×시간대 평균편차) + modelResidual(그날 기상조건별 추가보정)
                "refTemp": round(float(tmp_val), 1) if pd.notna(tmp_val) else None,
                "refHumidity": round(float(reh_val), 1) if pd.notna(reh_val) else None,
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

        # 코드가 "북한00-00" 형식이 아닌 예외(예: 족두리봉)는 수동 매핑으로 코드를 보정
        raw_name = st["다목적위치표지판번호"]
        code = CODE_ALIASES.get(raw_name, raw_name)
        detail_name = point_names_map.get(code)

        results.append({
            "id": st["국가지점번호"],
            "code": code,
            "name": code,
            "detailName": detail_name,
            "lat": st["GNSS-위도"],
            "lon": st["GNSS-경도"],
            "obs": obs_out,
            "forecasts": rows,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    extra_results = build_interpolated_points(args.extra_points, results, forecast, ultra_forecast)
    if extra_results:
        print(f"보간 추정 지점 {len(extra_results)}개 추가됨 (실측 없음, 주변 지점 IDW 보간)")
    all_results = results + extra_results

    # 원본 데이터 자체에 같은 코드(예: "둘레길105-02")가 서로 다른 실제 좌표의
    # 센서 2개에 중복 부여된 경우가 있어서, 화면에서 헷갈리지 않게 순번을 붙여 구분
    from collections import Counter
    code_counts = Counter(p["code"] for p in all_results)
    seen = Counter()
    for p in all_results:
        if code_counts[p["code"]] > 1:
            seen[p["code"]] += 1
            original = p["code"]
            p["code"] = f"{original}-{seen[original]}"
            p["name"] = p["code"]
            print(f"[안내] 코드 중복 발견, 구분자 추가: {original} -> {p['code']} (좌표 {p['lat']},{p['lon']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {args.out} (모델 예측 {len(results)}개 + 보간 추정 {len(extra_results)}개 = 총 {len(all_results)}개 지점)")

    if args.reference_areas_out:
        areas = build_reference_areas(current_obs)
        os.makedirs(os.path.dirname(args.reference_areas_out) or ".", exist_ok=True)
        with open(args.reference_areas_out, "w", encoding="utf-8") as f:
            json.dump(areas, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {args.reference_areas_out} (기준지역 {len(areas)}개)")


if __name__ == "__main__":
    main()
