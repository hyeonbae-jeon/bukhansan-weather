"""
기상청 기상특보통보문조회(getWthrWrnMsg)로 지금 전국에 발효 중인 특보를 받아와서,
북한산이 걸친 지역(종로구/강북구/은평구/성북구/고양시)에 해당하는 특보만 걸러 저장한다.

이 API는 전국 특보를 하나의 통보문 텍스트(t6 필드)로 묶어서 주기 때문에, 별도의
지역 필터 파라미터가 없다 — 텍스트 안에서 우리 지역명이 들어간 줄만 정규식으로 찾는다.

사용법:
    export KMA_WARN_API_KEY="발급받은 서비스키"
    python fetch_weather_warning.py --out ../data/weather/warnings.json

주의:
- 요청 파라미터 중 stnId/tmfc1/tmfc2가 정확히 필수인지 문서로 확인 못 해서, 우선
  최근 2일 범위 + 서울(108) 기준으로 요청하도록 짰다. 실행해서 안 되면(빈 결과거나
  에러) --debug 옵션으로 원본 응답을 보여달라.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from urllib.parse import unquote

import requests

BASE_URL = "http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnMsg"

# 북한산이 걸쳐있는 행정구역 — 이 이름이 특보 통보문 텍스트에 등장하면 우리 지역 특보로 판단
BUKHANSAN_DISTRICTS = ["종로", "강북", "은평", "성북", "고양", "도봉"]


def parse_warning_text(t6: str):
    """"o 강풍주의보 : 지역A, 지역B ... o 폭염경보 : 지역C ..." 형태 텍스트를
    특보종류별 지역목록으로 쪼갠 뒤, 북한산 관련 지역이 있는 것만 추린다."""
    if not t6:
        return []
    # "o 특보종류 : 내용" 단위로 분리
    entries = re.split(r"\s*o\s+", t6)
    matched = []
    for entry in entries:
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        warn_type, areas = entry.split(":", 1)
        warn_type = warn_type.strip()
        areas = areas.strip()
        hit_districts = [d for d in BUKHANSAN_DISTRICTS if d in areas]
        if hit_districts:
            matched.append({
                "type": warn_type,
                "matchedDistricts": hit_districts,
                "rawAreaText": areas[:200],  # 너무 길면 잘라서 저장
            })
    return matched


def fetch(service_key: str, debug: bool = False):
    now = datetime.now()
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "stnId": "108",
        "tmfc1": (now - timedelta(days=2)).strftime("%Y%m%d%H%M"),
        "tmfc2": now.strftime("%Y%m%d%H%M"),
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if debug:
        print("[디버그] 원본 응답(앞부분):", json.dumps(data, ensure_ascii=False)[:1500], file=sys.stderr)

    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})
    rows = items.get("item", []) if items else []
    if isinstance(rows, dict):
        rows = [rows]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    service_key = os.environ.get("KMA_WARN_API_KEY")
    if not service_key:
        print("환경변수 KMA_WARN_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)
    service_key = unquote(service_key)

    rows = fetch(service_key, debug=args.debug)
    if not rows:
        print("받아온 통보문이 없습니다 (특보가 없을 수도, 요청 파라미터가 안 맞을 수도 있어요).")
        result = {"updatedAt": datetime.now().isoformat(), "warnings": []}
    else:
        latest = rows[0]  # 가장 최근 통보문 하나만 사용
        matched = parse_warning_text(latest.get("t6", ""))
        result = {
            "updatedAt": datetime.now().isoformat(),
            "tmFc": latest.get("tmFc"),
            "tmEf": latest.get("tmEf"),
            "warnings": matched,
        }
        print(f"북한산 관련 특보 {len(matched)}건 발견")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
