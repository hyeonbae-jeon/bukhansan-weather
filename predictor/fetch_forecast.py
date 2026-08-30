"""
기상청 단기예보 조회서비스(getVilageFcst)로 향후 최대 3일치 예보를 받아온다.
우리 센서 지점들이 걸쳐있는 격자 셀(nx,ny)만 호출한다 (보통 몇 개 안 됨).

사용법:
    export KMA_API_KEY="발급받은 서비스키"
    python fetch_forecast.py --sensor ../data/sensor/sensor_merged.csv --out ../data/weather/forecast.csv
"""
import argparse
import concurrent.futures
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import requests

from http_retry import get_with_retry
import pandas as pd

from kma_grid import latlon_to_grid, grids_covering_geojson

BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
# GitHub Actions 실행기(ubuntu-latest)는 시스템 시간대가 UTC라, timezone 정보 없이
# datetime.now()를 쓰면 한국시간(KST, UTC+9)보다 9시간 뒤처진 "지금"이 나온다 — 반드시
# 명시적으로 한국시간 기준으로 계산해야 함 (fetch_current_obs.py 상단 설명 참고).
KST = timezone(timedelta(hours=9))


def latest_base_datetime(now: datetime):
    """단기예보는 하루 8회(02,05,08,11,14,17,20,23시) 발표되고, 발표 후 약 10분 뒤 제공된다.
    현재 시각 기준으로 '이미 발표되어 조회 가능한' 가장 최근 발표시각을 계산한다."""
    candidates = []
    for d in [now.date(), now.date() - timedelta(days=1)]:
        for t in BASE_TIMES:
            # now가 KST 등 timezone-aware라 candidate도 같은 tzinfo를 붙여야
            # 아래 "c + timedelta(...) <= now" 비교에서 naive/aware 섞여 에러 안 남
            dt = datetime.combine(d, datetime.strptime(t, "%H%M").time(), tzinfo=now.tzinfo)
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
                     help="국립공원 경계 GeoJSON 경로 (선택) - 주면 센서가 없는 구석까지 포함해서 "
                          "공원 전체를 덮는 격자를 다 가져옴(국지성 호우 대비, 기존엔 센서 있는 4개 격자만 조회했음)")
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

    # 격자 셀을 병렬로 조회 — 예전엔 순차 호출이라 한 셀이 재시도까지 다 실패하면
    # (최악의 경우 셀당 최대 약 3~4분) 그 시간만큼 그대로 늘어졌는데, 병렬로 바꾸면
    # 전체 소요시간이 "가장 오래 걸리는 셀 하나" 수준으로 줄어든다. 동시 요청 수는
    # max_workers로 제한해서 공공데이터포털 서버에 순간적으로 너무 몰리지 않게 한다.
    def _fetch_one(grid):
        nx, ny = grid
        try:
            return grid, fetch_grid_forecast(service_key, base_date, base_time, nx, ny), None
        except Exception as e:
            return grid, None, e

    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_fetch_one, g) for g in unique_grids]
        for future in concurrent.futures.as_completed(futures):
            (nx, ny), df, err = future.result()
            if err is not None:
                # 이 격자만 이번 회차에 건너뛴다 — 전체를 실패 처리하지 않고, 아래
                # "기존 파일과 병합" 로직 덕분에 이 격자의 직전 값이 자동으로 유지된다.
                print(f"  경고: 격자({nx},{ny}) 조회 실패(이 격자만 건너뜀, 기존 값 유지) - {err}",
                      file=sys.stderr)
                continue
            if df is None or df.empty:
                print(f"  경고: 격자({nx},{ny}) 응답 없음")
                continue
            print(f"[fetch] nx={nx}, ny={ny} 완료")
            all_rows.append(df)

    if not all_rows and not os.path.exists(args.out):
        print("받아온 예보가 없고 기존 파일도 없습니다.", file=sys.stderr)
        sys.exit(1)

    if all_rows:
        raw = pd.concat(all_rows, ignore_index=True)
        # category별로 세로로 쌓여있는 걸 가로(피벗)로 정리
        raw["fcstDateTime"] = pd.to_datetime(raw["fcstDate"] + raw["fcstTime"], format="%Y%m%d%H%M")
        pivot = raw.pivot_table(
            index=["grid", "fcstDateTime"], columns="category", values="fcstValue", aggfunc="first"
        ).reset_index()
    else:
        # 이번 회차에 격자가 전부 실패했어도, 기존 파일이 있으면 그걸 그대로 이어받아
        # 최소한 "직전 값"은 유지한 채로 계속 진행한다 (완전 실패로 스텝을 죽이지 않음).
        print("  경고: 이번 회차엔 격자를 하나도 못 받아왔음 - 기존 파일 값을 그대로 유지함", file=sys.stderr)
        pivot = pd.DataFrame(columns=["grid", "fcstDateTime"])

    # 기상청 단기예보 API는 "지금부터 미래"만 돌려주고 지나간 시각은 다시 안 줘서,
    # 매번 덮어쓰기만 하면 "몇 시간 전" 데이터가 사라져 지도의 과거 슬라이더가
    # 반쪽만 채워졌었다. 그래서 기존 파일을 이어붙여(누적) 과거 구간을 보존하고,
    # 최신 값(이번에 새로 받은 값)으로 겹치는 시각은 덮어쓴 뒤, 파일이 무한정
    # 커지지 않도록 [지금-30시간, 지금+80시간] 밖은 잘라낸다.
    if os.path.exists(args.out):
        try:
            old = pd.read_csv(args.out, parse_dates=["fcstDateTime"])
            combined = pd.concat([old, pivot], ignore_index=True)
            # keep="last": pivot(새로 받은 값)이 old보다 뒤에 있으니, 겹치는
            # (grid, fcstDateTime)은 새 값이 이긴다
            combined = combined.drop_duplicates(subset=["grid", "fcstDateTime"], keep="last")
        except Exception as e:
            print(f"  경고: 기존 파일을 못 읽어서 새로 받은 값만 씀 - {e}", file=sys.stderr)
            combined = pivot
    else:
        combined = pivot

    now_naive = now.replace(tzinfo=None)  # fcstDateTime은 타임존 정보 없는 한국시간 기준
    window_start = now_naive - timedelta(hours=30)
    window_end = now_naive + timedelta(hours=80)
    combined = combined[(combined["fcstDateTime"] >= window_start) & (combined["fcstDateTime"] <= window_end)]
    combined = combined.sort_values(["grid", "fcstDateTime"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(combined)}행, {combined['grid'].nunique()}개 격자, 누적 기간 {window_start}~{window_end})")


if __name__ == "__main__":
    main()
