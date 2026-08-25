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
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import requests

from http_retry import get_with_retry
import pandas as pd

from kma_grid import latlon_to_grid, grids_covering_geojson

BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
# GitHub Actions 실행기(ubuntu-latest)는 시스템 시간대가 UTC라, timezone 정보 없이
# datetime.now()를 쓰면 한국시간(KST, UTC+9)보다 9시간 뒤처진 값이 나온다. 그 어긋난
# "지금"으로 기준시각을 계산해도 KMA API 요청 자체는 (그 과거 시각의 자료가 실제로
# 있으니) 에러 없이 성공해버려서, 파이프라인은 매번 정상 종료되는데 관측값은 계속
# 9~10시간 전 것만 갱신되는 상황이 생겼었다. 반드시 한국시간 기준으로 계산해야 함.
KST = timezone(timedelta(hours=9))


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
