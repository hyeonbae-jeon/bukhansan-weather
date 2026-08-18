"""
기상청 지상(종관, ASOS) 시간자료 조회서비스로 과거 시간별 관측 데이터를 받아온다.
기준 관측소: 108 (서울, 종로구 송월동) — 북한산에서 가장 가까운 ASOS 관측소.

사용법:
    export KMA_API_KEY="발급받은 서비스키(디코딩 안 된 그대로 또는 인코딩 키 아무거나 - 아래 참고)"
    python fetch_kma_hourly.py --start 20250101 --end 20260531 --out ../data/weather/asos_108.csv

주의:
- data.go.kr에서 발급하는 서비스키는 '일반 인증키(Encoding)'와 '일반 인증키(Decoding)' 두 종류가 있다.
  requests가 자체적으로 쿼리스트링을 인코딩하므로, 여기서는 Decoding 키를 쓰는 걸 권장한다.
  (Encoding 키를 쓰면 %가 이중 인코딩되어 인증 오류가 날 수 있다.)
"""
import argparse
import os
import time
import sys
from datetime import datetime, timedelta

import requests
import pandas as pd

BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
STATION_ID = "108"  # 서울(종로구 송월동)


def fetch_page(service_key: str, start_dt: str, end_dt: str, page_no: int, num_of_rows: int = 999):
    params = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_dt,
        "startHh": "00",
        "endDt": end_dt,
        "endHh": "23",
        "stnIds": STATION_ID,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    try:
        body = resp.json()["response"]["body"]
    except Exception:
        print("응답 파싱 실패, 원본 응답:", resp.text[:500], file=sys.stderr)
        raise
    return body


def fetch_range(service_key: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    """start_dt~end_dt(YYYYMMDD)를 한 번에 요청 범위가 너무 길면 나눠서 호출한다.
    ASOS 시간자료는 기간이 길어도 되지만 안전하게 3개월 단위로 끊어서 호출한다."""
    all_rows = []
    cur = datetime.strptime(start_dt, "%Y%m%d")
    end = datetime.strptime(end_dt, "%Y%m%d")

    while cur <= end:
        chunk_end = min(cur + timedelta(days=90), end)
        s = cur.strftime("%Y%m%d")
        e = chunk_end.strftime("%Y%m%d")
        print(f"[fetch] {s} ~ {e}")

        page_no = 1
        while True:
            body = fetch_page(service_key, s, e, page_no)
            total = int(body.get("totalCount", 0))
            items = body.get("items", {})
            rows = items.get("item", []) if items else []
            if isinstance(rows, dict):
                rows = [rows]
            all_rows.extend(rows)

            fetched_so_far = page_no * 999
            if fetched_so_far >= total or not rows:
                break
            page_no += 1
            time.sleep(0.2)

        cur = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    service_key = os.environ.get("KMA_API_KEY")
    if not service_key:
        print("환경변수 KMA_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    df = fetch_range(service_key, args.start, args.end)
    if df.empty:
        print("받아온 데이터가 없습니다. 인증키/기간을 확인하세요.", file=sys.stderr)
        sys.exit(1)

    # 필요한 컬럼만 정리 (tm=관측시각, ta=기온, hm=습도, rn=강수량, ws=풍속, wd=풍향, ca_tot=전운량)
    keep_cols = {
        "tm": "관측시각",
        "ta": "기온",
        "hm": "습도",
        "rn": "강수량",
        "ws": "풍속",
        "wd": "풍향",
        "ca_tot": "전운량",
    }
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].rename(columns=keep_cols)
    df["관측시각"] = pd.to_datetime(df["관측시각"])
    for c in ["기온", "습도", "강수량", "풍속", "풍향", "전운량"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(df)}행)")


if __name__ == "__main__":
    main()
