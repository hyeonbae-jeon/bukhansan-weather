"""
기상청 초단기실황조회(getUltraSrtNcst)로 "지금 실제로 관측된" 날씨를 가져온다.
예보(forecast)가 아니라 실황(observation)이라 이게 진짜 "지금 비가 오는지"에 대한
가장 정확한 답이다. 별도 API 신청 없이 기존 "단기예보 조회서비스" 키로 바로 호출된다
(같은 서비스 그룹에 초단기실황/초단기예보/단기예보가 함께 묶여있음).

우리 센서 지점들이 걸쳐있는 격자 셀(보통 4개)만 호출한다.

사용법:
    export KMA_API_KEY="발급받은 서비스키"
    python fetch_current_obs.py --sensor ../data/sensor/sensor_merged.csv --out ../data/weather/current_obs.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import unquote

import requests

from http_retry import get_with_retry
import pandas as pd

from kma_grid import latlon_to_grid

BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"


def latest_base_datetime(now: datetime):
    """초단기실황은 매시 정각 관측치를 40분 뒤에 제공한다. 여유를 두고 45분을 기준으로
    이번 시각 걸 쓸지 직전 시각 걸 쓸지 정한다."""
    if now.minute < 45:
        base = now - timedelta(hours=1)
    else:
        base = now
    base = base.replace(minute=0, second=0, microsecond=0)
    return base.strftime("%Y%m%d"), base.strftime("%H%M")


def fetch_grid_obs(service_key: str, base_date: str, base_time: str, nx: int, ny: int) -> dict:
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 20,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    resp = get_with_retry(BASE_URL, params, timeout=30)
    body = resp.json()["response"]["body"]
    items = body.get("items", {})
    rows = items.get("item", []) if items else []
    if isinstance(rows, dict):
        rows = [rows]

    result = {"baseDateTime": f"{base_date} {base_time}"}
    for row in rows:
        category = row.get("category")
        value = row.get("obsrValue")
        if category in ("T1H", "RN1", "REH", "PTY", "WSD", "VEC"):
            try:
                result[category] = float(value)
            except (TypeError, ValueError):
                result[category] = value
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    service_key = os.environ.get("KMA_API_KEY")
    if not service_key:
        print("환경변수 KMA_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)
    service_key = unquote(service_key)

    sensor = pd.read_csv(args.sensor)
    points = sensor[["국가지점번호", "GNSS-경도", "GNSS-위도"]].drop_duplicates("국가지점번호")
    points["grid"] = points.apply(
        lambda r: latlon_to_grid(r["GNSS-위도"], r["GNSS-경도"]), axis=1
    )
    unique_grids = points["grid"].unique()
    print(f"고유 격자 셀 {len(unique_grids)}개: {list(unique_grids)}")

    now = datetime.now()
    base_date, base_time = latest_base_datetime(now)
    print(f"기준 관측시각: {base_date} {base_time}")

    result = {}
    for nx, ny in unique_grids:
        key = f"{nx}_{ny}"
        print(f"[fetch] {key}")
        try:
            result[key] = fetch_grid_obs(service_key, base_date, base_time, nx, ny)
        except Exception as e:
            print(f"  경고: {key} 조회 실패 - {e}", file=sys.stderr)
        time.sleep(0.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
