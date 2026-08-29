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
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

from kma_grid import latlon_to_grid

# GitHub Actions 실행기(ubuntu-latest)는 시스템 시간대가 UTC라, timezone 정보 없이
# datetime.now()를 쓰면 한국시간(KST, UTC+9)보다 9시간 뒤처진 값이 나온다 (다른
# fetch_*.py 스크립트에서 이미 겪은 문제와 동일한 이유로 명시적으로 KST를 씀).
KST = timezone(timedelta(hours=9))

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


def apply_obs_to_row(row, pty, rn1):
    """실황 관측값(pty, rn1)으로 예측 행 하나(딕셔너리)를 덮어쓴다. 예보와 달리
    실황은 "지금/그때 실제로 관측된 값"이라 방향을 가릴 이유가 없어서(뒤에 나오는
    AWS 비대칭 보정과 다름) 강수 있음/없음 양방향 다 반영한다. pty가 None이면
    (그 시각 관측값이 아예 없으면) 아무 것도 안 하고 False를 돌려준다."""
    if pty is None:
        return False
    pty_int = int(pty)
    row["pty"] = pty_int
    if pty_int == 0:
        row["precipMm"] = None
        row["precipLabel"] = None
    elif pty_int not in (3, 7):  # 비 계열일 때만 실황 강수량 반영(눈은 실황에 적설량 없음)
        precip_mm, precip_label = parse_precip_amount(rn1, "mm")
        if precip_mm is not None or precip_label:
            row["precipMm"] = precip_mm
            row["precipUnit"] = "mm"
            row["precipLabel"] = precip_label
    return True


def hour_key_of(time_str):
    """예측 행의 "YYYY-MM-DDTHH:MM" 시각을 obs_history의 정시 키("YYYY-MM-DDTHH:00")로 맞춘다."""
    return time_str[:13] + ":00"


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


def build_interpolated_points(extra_points_csv, real_results, forecast=None, ultra_forecast=None, current_obs=None, obs_history=None):
    """온습도 센서가 없는 지점들을, 실측 기반 모델 예측이 있는 주변 지점들로부터
    역거리가중(IDW)으로 보간해서 만든다. 모델 예측이 아니라 '추정치'임을
    interpolated:true로 명시해서 프론트에서 구분 표시할 수 있게 한다.

    강수형태/하늘상태/강수확률/강수량은 (예전엔 IDW 루프에서 "가장 먼저 만난 실측
    지점" 값을 그냥 갖다 썼는데, 그게 거리와 무관하게 리스트 순서상 우연히 먼저
    나온 지점이라 엉뚱한(먼) 지점의 값을 쓰는 경우가 있었다) 이제 이 지점 자신의
    좌표로 계산한 진짜 자기 격자에서 직접 가져온다 — 기온/습도만 IDW로 보간하고
    강수 관련 값은 실측 지점과 무관하게 그 위치의 실제 격자 예보를 그대로 씀.

    기온 보간에는 표준 기온감률(고도 100m당 약 0.65도) 보정을 더한다. 원래는
    수평 거리 기반 IDW만 써서, 가까이 있어도 고도가 많이 다른 지점끼리(예:
    능선 위 vs 계곡) 기온이 부자연스럽게 똑같이 나오는 문제가 있었음. 이제
    실측 센서 하나하나의 진짜 고도(point_elevations.csv로 매칭됨)를 알고 있어서,
    "이웃마다" 그 이웃과 보간지점 사이의 고도차만큼 먼저 보정한 값으로 IDW를
    한다(예: 이웃이 나보다 200m 높으면 그 이웃 기온에 +1.3도 해서 평균에 반영)."""
    if not extra_points_csv or not os.path.exists(extra_points_csv):
        return []

    extra_df = pd.read_csv(extra_points_csv)
    if extra_df.empty:
        return []

    LAPSE_RATE_C_PER_100M = 0.65  # 대류권 평균 기온감률(표준대기)
    # 이웃 실측지점 고도를 못 구한 경우(예: point_elevations.csv 자체가 없을 때)의
    # 최후 대안으로만 쓰는 전체 평균 — 정상적으로는 아래 개별 고도 보정이 우선 적용됨
    mean_elevation = extra_df["elevation_m"].mean() if "elevation_m" in extra_df.columns else None

    # 모든 지점이 공유하는 예보 시각 목록 (첫 실측 지점 기준)
    times = [f["time"] for f in real_results[0]["forecasts"]] if real_results else []

    results = []
    for _, ep in extra_df.iterrows():
        ep_grid = "{}_{}".format(*latlon_to_grid(ep["lat"], ep["lon"]))
        ep_elevation = ep.get("elevation_m")

        forecasts = []
        for t in times:
            num_t, den_t, num_h, den_h = 0.0, 0.0, 0.0, 0.0
            for rp in real_results:
                f = next((x for x in rp["forecasts"] if x["time"] == t), None)
                if not f or f["temp"] is None:
                    continue
                d = haversine_km(ep["lat"], ep["lon"], rp["lat"], rp["lon"])

                # 이 이웃의 기온을, "이 이웃이 보간지점과 같은 고도에 있었다면
                # 얼마였을지"로 먼저 보정한 뒤에 거리 가중 평균에 넣는다
                rp_elevation = rp.get("elevationM")
                temp_val = f["temp"]
                if pd.notna(ep_elevation) and rp_elevation is not None and pd.notna(rp_elevation):
                    temp_val = temp_val + LAPSE_RATE_C_PER_100M * (rp_elevation - ep_elevation) / 100

                if d < 0.01:
                    num_t, den_t = temp_val, 1
                    num_h, den_h = (f["humidity"] or 0), 1
                    break
                w = 1 / (d ** 2)
                num_t += w * temp_val
                den_t += w
                if f["humidity"] is not None:
                    num_h += w * f["humidity"]
                    den_h += w

            temp = round(num_t / den_t, 1) if den_t > 0 else None
            # 이웃들 중 고도 정보가 하나도 없었던 경우(전부 위 보정을 못 받음)에만
            # "전체 평균 대비" 근사 보정을 최후 수단으로 적용
            if (temp is not None and mean_elevation is not None and pd.notna(ep_elevation)
                    and not any(rp.get("elevationM") is not None for rp in real_results)):
                elev_diff = ep_elevation - mean_elevation
                temp = round(temp - LAPSE_RATE_C_PER_100M * elev_diff / 100, 1)
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

    # "지금" 행을 실황(있으면)으로 덮어쓴다 — 실측 센서가 있는 지점과 똑같은 이유
    # (초단기예보는 매시 한 번만 갱신되는 "예보"라 비가 그쳐도/시작해도 다음 갱신
    # 전까진 옛날 값이 남아있음). 보간 지점도 자기 격자의 실황을 그대로 쓴다.
    now_ts = pd.Timestamp(datetime.now(KST).replace(tzinfo=None))
    if current_obs:
        for r in results:
            ep_grid = "{}_{}".format(*latlon_to_grid(r["lat"], r["lon"]))
            obs_entry = current_obs.get(ep_grid)
            obs_pty = obs_entry.get("PTY") if obs_entry else None
            if obs_pty is None or not r["forecasts"]:
                continue
            now_row = min(r["forecasts"], key=lambda f: abs(pd.Timestamp(f["time"]) - now_ts))
            apply_obs_to_row(now_row, obs_pty, obs_entry.get("RN1"))

    # 이미 지나간 시각들도 쌓아둔 실황 기록(obs_history)으로 보정 — 실측 센서가
    # 있는 지점과 같은 이유(위 주석 참고), API 추가 호출 없이 기존 기록만 재사용.
    if obs_history:
        for r in results:
            if not r["forecasts"]:
                continue
            ep_grid = "{}_{}".format(*latlon_to_grid(r["lat"], r["lon"]))
            grid_hist = obs_history.get(ep_grid)
            if not grid_hist:
                continue
            for row in r["forecasts"]:
                if pd.Timestamp(row["time"]) >= now_ts:
                    continue
                hist_entry = grid_hist.get(hour_key_of(row["time"]))
                if hist_entry:
                    apply_obs_to_row(row, hist_entry.get("PTY"), hist_entry.get("RN1"))

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

# 방재기상관측(AWS) 실측 지점 — 격자 대표값이 아니라 실제 관측소 원본값이라 더
# 정확함(fetch_aws_obs.py 참고). "기상청 관측값" 지역 목록에 넣는 대신(사용자 요청
# 으로 뺌), 각 관측소 반경 안의 지점들의 "지금" 강수 여부를 실측대로 보정하는 데
# 쓴다 — 원래 AWS를 넣은 목적이 격자보다 촘촘한 강수 구역 파악이었으므로.
AWS_STATIONS = [
    # 좌표는 기상청 API허브 지점정보(stn_inf.php?inf=AWS) 공식 값으로 교체함 —
    # 예전 좌표(스크린샷 등으로 대략 확보한 값)와 비교해보니 424(강북)는 2.8km,
    # 416(은평)은 4.2km, 540(고양)은 6.6km나 차이가 났다. 보정 반경이 2.5km인데
    # 이보다 좌표 오차가 더 커서, 실제로는 그 관측소 근처가 아닌 엉뚱한 위치를
    # 기준으로 "반경 2.5km 이내 지점"을 골라내고 있었던 셈 — 이번에 정확한 값으로
    # 바로잡음. "420 북한산"은 여전히 공식 지점 목록에 없는 존재하지 않는
    # 번호라 빼뒀다(예전에 확인된 문제, 그대로 유지).
    {"stn": "424", "name": "강북", "lat": 37.63801, "lon": 127.00981},
    {"stn": "416", "name": "은평", "lat": 37.64647, "lon": 126.94273},
    {"stn": "414", "name": "성북", "lat": 37.61172, "lon": 126.99944},
    {"stn": "540", "name": "고양", "lat": 37.63730, "lon": 126.89200},
    # 공식 지점 목록을 보니 공원 북동쪽(도봉산/사패산 쪽)을 보정해줄 관측소가
    # 없었는데, "도봉"(406)이 그 방향에 정확히 있어서 새로 추가함.
    {"stn": "406", "name": "도봉", "lat": 37.66612, "lon": 127.02947},
]
AWS_CORRECTION_RADIUS_KM = 2.5  # 이 반경 안의 지점만 AWS 실측으로 "지금" 강수를 보정


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


def apply_aws_rain_correction(all_results: list, aws_obs: dict | None) -> int:
    """AWS 관측소에서 반경 AWS_CORRECTION_RADIUS_KM 이내에 있는 지점들의, "지금"에
    가장 가까운 예보 시각 하나만 골라 강수 여부를 실측대로 보정한다.

    비대칭적으로 적용함: AWS가 "비 온다"고 하면 그 근처 지점들의 강수 없음(pty=0)
    예보를 실측대로 덮어써서 놓친 소나기를 잡아내고, 반대로 AWS가 "비 없다"고
    해도 예보가 이미 강수를 예상했으면 그대로 둔다(이동 중인 비구름이 아직
    관측소엔 안 닿았을 수도 있어서, 예보의 강수 경고를 섣불리 지우지 않음 —
    안전 쪽으로 치우치는 게 낫다는 판단)."""
    if not aws_obs or not all_results:
        return 0

    stations = []
    for area in AWS_STATIONS:
        entry = aws_obs.get(area["stn"])
        if not entry:
            continue
        rain1h = entry.get("rain1h")
        temp = entry.get("temp")
        if not (rain1h and rain1h > 0):
            continue
        pty = 3 if (temp is not None and temp <= 0) else 1  # 0도 이하면 눈으로 대략 구분
        stations.append({**area, "pty": pty, "rain1h": rain1h})
    if not stations:
        return 0

    now = datetime.now(KST).replace(tzinfo=None)
    times = [f["time"] for f in all_results[0]["forecasts"]]
    now_time = min(times, key=lambda t: abs(pd.Timestamp(t) - now))

    corrected = 0
    for p in all_results:
        f_now = next((x for x in p["forecasts"] if x["time"] == now_time), None)
        if not f_now or f_now.get("pty"):
            continue  # 이미 강수 예보가 있으면 안 건드림(비대칭 보정)
        for st in stations:
            if haversine_km(p["lat"], p["lon"], st["lat"], st["lon"]) <= AWS_CORRECTION_RADIUS_KM:
                f_now["pty"] = st["pty"]
                if not f_now.get("precipMm"):
                    f_now["precipMm"] = st["rain1h"]
                    f_now["precipUnit"] = "mm"
                    f_now["precipLabel"] = f"AWS({st['name']}) 실측 {st['rain1h']}mm"
                corrected += 1
                break
    return corrected


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
    ap.add_argument("--obs-history", required=False, default=None,
                     help="fetch_current_obs.py --history-out으로 계속 쌓아온 시간대별 실황 기록 "
                          "경로 (선택). 있으면 '지금'뿐 아니라 이미 지나간 시각들도 그때의 실황으로 "
                          "보정한다(과거 24시간 구간). 없으면 지금까지처럼 '지금' 한 칸만 보정됨.")
    ap.add_argument("--point-names", required=False, default=None,
                     help="지점 코드->세부지명 매핑 CSV (id,name) - 선택, data/point_names.csv")
    ap.add_argument("--reference-areas-out", required=False, default=None,
                     help="'기상청 관측값' 지역 선택용 별도 JSON 저장 경로 (선택, 예: ../frontend/reference_areas.json)")
    ap.add_argument("--ultra-forecast", required=False, default=None,
                     help="fetch_ultra_forecast.py가 만든 6시간 이내 초단기예보 CSV 경로 (선택) - "
                          "있으면 겹치는 시각의 기온/습도/강수형태/하늘상태/바람을 이걸로 우선 대체")
    ap.add_argument("--aws-obs", required=False, default=None,
                     help="fetch_aws_obs.py가 만든 AWS 실측 관측소 JSON 경로 (선택) - "
                          "있으면 가까운 지점들의 '지금' 강수 여부를 이 실측값으로 보정함")
    ap.add_argument("--elevations", required=False, default=None,
                     help="point_elevations.csv 경로 (선택, 실측 센서 120개의 고도) - "
                          "있으면 보간 지점 기온 보정에 이웃별 정확한 고도차를 씀")
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

    # 과거 구간 보정용 — 실패해도(파일 손상 등) 조용히 빈 상태로 넘어가고 나머지
    # 파이프라인(현재 예측 생성, 커밋)에는 전혀 지장 없게 한다.
    obs_history = {}
    if args.obs_history and os.path.exists(args.obs_history):
        try:
            with open(args.obs_history, encoding="utf-8") as f:
                obs_history = json.load(f)
            print(f"실황 기록(과거 보정용) 로드됨: {len(obs_history)}개 격자")
        except Exception as e:
            print(f"[경고] 실황 기록 로드 실패(과거 보정 없이 진행) - {e}")

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

    # 실측 센서 120개의 고도 — 예전엔 station_meta.json에 고도가 아예 없어서 보간
    # 지점(75개, 자체 고도 있음)의 기온 보정이 "동네 평균 고도" 근사치로만 가능했는데,
    # 이제 지점 마스터 목록(point_elevations.csv, 187개 전체 고도 포함)에서 실측
    # 센서 하나하나의 진짜 고도를 매칭해서 훨씬 정밀한 지점별 기온감률 보정이 가능함.
    station_elevations = {}
    if args.elevations and os.path.exists(args.elevations):
        elev_df = pd.read_csv(args.elevations)
        station_elevations = dict(zip(elev_df["national_id"], elev_df["elevation_m"]))
        matched = stations["국가지점번호"].isin(station_elevations).sum()
        print(f"실측 센서 고도 매칭됨: {matched}/{len(stations)}개 (point_elevations.csv 기준)")
    else:
        print("실측 센서 고도 파일 없음 (--elevations 미지정) — 보간 지점 기온 보정이 부정확할 수 있음")

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

            # "지금"에 가장 가까운 예보 행을 실황으로 덮어쓴다. 초단기예보(PTY의
            # 기본 출처)는 매시 한 번만 갱신되는 "예보"라서, 그 사이에 비가
            # 그쳐도(또는 갑자기 시작해도) 다음 갱신 전까지 옛날 값을 계속
            # 보여주는 문제가 있었다 — 날씨누리는 실황을 바로 반영해서 더 빨리
            # 사라지는데 우리 사이트엔 강수가 남아있던 원인이 이거였음.
            if rows and obs_pty is not None:
                now_ts = pd.Timestamp(datetime.now(KST).replace(tzinfo=None))
                now_row = min(rows, key=lambda r: abs(pd.Timestamp(r["time"]) - now_ts))
                apply_obs_to_row(now_row, obs_pty, obs_entry.get("RN1"))

        # 이미 지나간 시각들도, 그때 쌓아둔 실황 기록(obs_history)이 있으면 실황으로
        # 보정한다 — API를 새로 부르지 않고 fetch_current_obs.py가 매 실행마다
        # 이미 쌓아둔 기록만 재사용하므로 추가 호출·추가 실패 지점이 없다.
        if obs_history and rows:
            grid_hist = obs_history.get(st["grid"])
            if grid_hist:
                now_ts = pd.Timestamp(datetime.now(KST).replace(tzinfo=None))
                for row in rows:
                    if pd.Timestamp(row["time"]) >= now_ts:
                        continue  # 미래 시각은 실황으로 손대지 않음(관측값이 없으므로 당연)
                    hist_entry = grid_hist.get(hour_key_of(row["time"]))
                    if hist_entry:
                        apply_obs_to_row(row, hist_entry.get("PTY"), hist_entry.get("RN1"))

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
            "elevationM": station_elevations.get(st["국가지점번호"]),
            "obs": obs_out,
            "forecasts": rows,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    extra_results = build_interpolated_points(args.extra_points, results, forecast, ultra_forecast, current_obs, obs_history)
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

    aws_obs = None
    if args.aws_obs and os.path.exists(args.aws_obs):
        with open(args.aws_obs, encoding="utf-8") as f:
            aws_obs = json.load(f)
        print(f"AWS 실측 관측소 로드됨: {len(aws_obs)}개")
        corrected_n = apply_aws_rain_correction(all_results, aws_obs)
        if corrected_n:
            print(f"AWS 실측 기준으로 '지금' 강수를 보정한 지점 {corrected_n}개 (반경 {AWS_CORRECTION_RADIUS_KM}km 이내)")

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
