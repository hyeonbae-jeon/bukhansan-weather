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
import concurrent.futures
import os
import sys
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

    # 격자 셀 병렬 조회 (fetch_forecast.py와 동일한 이유 — 순차 호출 시 한 셀의
    # 재시도 실패가 그대로 전체 소요시간에 더해짐)
    def _fetch_one(grid):
        nx, ny = grid
        try:
            return grid, fetch_grid_ultra_forecast(service_key, base_date, base_time, nx, ny), None
        except Exception as e:
            return grid, None, e

    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_fetch_one, g) for g in unique_grids]
        for future in concurrent.futures.as_completed(futures):
            (nx, ny), df, err = future.result()
            if err is not None:
                print(f"  경고: 격자({nx},{ny}) 조회 실패(이 격자만 건너뜀, 기존 값 유지) - {err}",
                      file=sys.stderr)
                continue
            if df is None or df.empty:
                print(f"  경고: 격자({nx},{ny}) 응답 없음")
                continue
            print(f"[fetch] nx={nx}, ny={ny} 완료")
            all_rows.append(df)

    if not all_rows and not os.path.exists(args.out):
        print("받아온 초단기예보가 없고 기존 파일도 없습니다.", file=sys.stderr)
        sys.exit(1)

    if all_rows:
        raw = pd.concat(all_rows, ignore_index=True)
        raw["fcstDateTime"] = pd.to_datetime(raw["fcstDate"] + raw["fcstTime"], format="%Y%m%d%H%M")
        pivot = raw.pivot_table(
            index=["grid", "fcstDateTime"], columns="category", values="fcstValue", aggfunc="first"
        ).reset_index()
    else:
        print("  경고: 이번 회차엔 격자를 하나도 못 받아왔음 - 기존 파일 값을 그대로 유지함", file=sys.stderr)
        pivot = pd.DataFrame(columns=["grid", "fcstDateTime"])

    # 이 파일은 원래 "지금부터 6시간"짜리 최신 스냅샷만 덮어쓰는 방식이었는데, 그러면
    # 이번 회차에 실패한 격자는 통째로 사라져버린다. 그래서 fetch_forecast.py처럼
    # 기존 파일과 병합(새 값이 있으면 덮어쓰고, 없으면 직전 값 유지)한 뒤, 초단기예보
    # 성격에 안 맞는 오래된 시각(2시간 넘게 지난 과거)만 정리한다.
    if os.path.exists(args.out):
        try:
            old = pd.read_csv(args.out, parse_dates=["fcstDateTime"])
            combined = pd.concat([old, pivot], ignore_index=True)
            combined = combined.drop_duplicates(subset=["grid", "fcstDateTime"], keep="last")
        except Exception as e:
            print(f"  경고: 기존 파일을 못 읽어서 새로 받은 값만 씀 - {e}", file=sys.stderr)
            combined = pivot
    else:
        combined = pivot

    now_naive = now.replace(tzinfo=None)
    window_start = now_naive - timedelta(hours=2)
    combined = combined[combined["fcstDateTime"] >= window_start]
    combined = combined.sort_values(["grid", "fcstDateTime"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(combined)}행, {combined['grid'].nunique()}개 격자)")


if __name__ == "__main__":
    main()
