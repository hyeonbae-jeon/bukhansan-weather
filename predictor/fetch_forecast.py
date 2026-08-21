"""
기상청 단기예보 조회서비스(getVilageFcst)로 향후 최대 3일치 예보를 받아온다.
우리 센서 지점들이 걸쳐있는 격자 셀(nx,ny)만 호출한다 (보통 몇 개 안 됨).

사용법:
    export KMA_API_KEY="발급받은 서비스키"
    python fetch_forecast.py --sensor ../data/sensor/sensor_merged.csv --out ../data/weather/forecast.csv
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import unquote

import requests
import pandas as pd

from kma_grid import latlon_to_grid

BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]


def latest_base_datetime(now: datetime):
    """단기예보는 하루 8회(02,05,08,11,14,17,20,23시) 발표되고, 발표 후 약 10분 뒤 제공된다.
    현재 시각 기준으로 '이미 발표되어 조회 가능한' 가장 최근 발표시각을 계산한다."""
    candidates = []
    for d in [now.date(), now.date() - timedelta(days=1)]:
        for t in BASE_TIMES:
            dt = datetime.combine(d, datetime.strptime(t, "%H%M").time())
            candidates.append(dt)
    candidates = sorted(candidates)
    usable = [c for c in candidates if c + timedelta(minutes=10) <= now]
    chosen = usable[-1]
    return chosen.strftime("%Y%m%d"), chosen.strftime("%H%M")


def fetch_grid_forecast(service_key: str, base_date: str, base_time: str, nx: int, ny: int) -> pd.DataFrame:
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()["response"]["body"]
    items = body.get("items", {})
    rows = items.get("item", []) if items else []
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["grid"] = f"{nx}_{ny}"
    return df


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
    print(f"기준 발표시각: {base_date} {base_time}")

    all_rows = []
    for nx, ny in unique_grids:
        print(f"[fetch] nx={nx}, ny={ny}")
        df = fetch_grid_forecast(service_key, base_date, base_time, nx, ny)
        if df.empty:
            print(f"  경고: 격자({nx},{ny}) 응답 없음")
            continue
        all_rows.append(df)
        time.sleep(0.3)

    if not all_rows:
        print("받아온 예보가 없습니다.", file=sys.stderr)
        sys.exit(1)

    raw = pd.concat(all_rows, ignore_index=True)
    # category별로 세로로 쌓여있는 걸 가로(피벗)로 정리
    raw["fcstDateTime"] = pd.to_datetime(raw["fcstDate"] + raw["fcstTime"], format="%Y%m%d%H%M")
    pivot = raw.pivot_table(
        index=["grid", "fcstDateTime"], columns="category", values="fcstValue", aggfunc="first"
    ).reset_index()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pivot.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(pivot)}행, {pivot['grid'].nunique()}개 격자)")


if __name__ == "__main__":
    main()
