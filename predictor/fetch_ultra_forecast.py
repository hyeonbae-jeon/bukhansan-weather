"""
기상청 초단기예보 조회서비스(getUltraSrtFcst)로 지금부터 6시간 뒤까지의 예보를 받아온다.

왜 필요한가:
단기예보(getVilageFcst, fetch_forecast.py)는 하루 8번(3시간 간격)만 발표돼서, 소나기처럼
짧은 시간에 생겼다 없어지는 위험기상을 놓치기 쉽다. 초단기예보는 "이 문제(짧은 시간에
발생·소멸하는 위험기상 대응)"를 위해 기상청이 만든 상품으로, 매시 정각+30분에 새로
발표되고 향후 6시간을 1시간 단위로 준다 — 단기예보보다 3배 자주 갱신되고, 실제로 비가
오기 시작하면 훨씬 빨리 반영된다.

사용법:
    export KMA_API_KEY="발급받은 서비스키"
    python fetch_ultra_forecast.py --sensor ../data/sensor/sensor_merged.csv --out ../data/weather/ultra_forecast.csv
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pandas as pd

from http_retry import get_with_retry
from kma_grid import latlon_to_grid, grids_covering_geojson

BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
# GitHub Actions 실행기(ubuntu-latest)는 시스템 시간대가 UTC라, timezone 정보 없이
# datetime.now()를 쓰면 한국시간(KST, UTC+9)보다 9시간 뒤처진 "지금"이 나온다.
KST = timezone(timedelta(hours=9))


def latest_base_datetime(now: datetime):
    """초단기예보는 매시 30분에 발표되고 10분 뒤(40분경)부터 조회 가능하다. 안전하게
    45분을 기준으로 이번 시각(HH30) 걸 쓸지 직전 시각(HH-1:30) 걸 쓸지 정한다."""
    if now.minute < 45:
        base = (now - timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)
    else:
        base = now.replace(minute=30, second=0, microsecond=0)
    return base.strftime("%Y%m%d"), base.strftime("%H%M")


def fetch_grid_ultra_forecast(service_key: str, base_date: str, base_time: str, nx: int, ny: int) -> pd.DataFrame:
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 200,
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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["grid"] = f"{nx}_{ny}"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundary", required=False, default=None,
                     help="국립공원 경계 GeoJSON 경로 (선택) - 공원 전체를 덮는 격자를 다 가져와서 "
                          "국지성 호우 대비 해상도를 높임 (fetch_forecast.py와 동일한 이유)")
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
    sensor_grids = set(points["grid"].unique())

    boundary_grids = set()
    if args.boundary and os.path.exists(args.boundary):
        boundary_grids = set(grids_covering_geojson(args.boundary))
        print(f"공원 경계가 덮는 격자 {len(boundary_grids)}개 추가 확인됨")

    unique_grids = sorted(sensor_grids | boundary_grids)
    print(f"조회할 격자 셀 {len(unique_grids)}개: {unique_grids}")

    now = datetime.now(KST)
    base_date, base_time = latest_base_datetime(now)
    print(f"기준 발표시각: {base_date} {base_time}")

    all_rows = []
    for nx, ny in unique_grids:
        print(f"[fetch] nx={nx}, ny={ny}")
        df = fetch_grid_ultra_forecast(service_key, base_date, base_time, nx, ny)
        if df.empty:
            print(f"  경고: 격자({nx},{ny}) 응답 없음")
            continue
        all_rows.append(df)
        time.sleep(0.3)

    if not all_rows:
        print("받아온 초단기예보가 없습니다.", file=sys.stderr)
        sys.exit(1)

    raw = pd.concat(all_rows, ignore_index=True)
    raw["fcstDateTime"] = pd.to_datetime(raw["fcstDate"] + raw["fcstTime"], format="%Y%m%d%H%M")
    pivot = raw.pivot_table(
        index=["grid", "fcstDateTime"], columns="category", values="fcstValue", aggfunc="first"
    ).reset_index()

    # 이 파일은 항상 "지금부터 6시간"짜리 최신 스냅샷만 있으면 되고(지나간 시각은
    # generate_predictions.py가 어차피 안 씀), 단기예보처럼 과거를 누적 보존할 필요가
    # 없어서 매번 그냥 덮어쓴다.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pivot.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(pivot)}행, {pivot['grid'].nunique()}개 격자)")


if __name__ == "__main__":
    main()
