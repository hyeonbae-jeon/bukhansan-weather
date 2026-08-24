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
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import requests

from http_retry import get_with_retry
import pandas as pd

from kma_grid import latlon_to_grid

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

    now = datetime.now(KST)
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
