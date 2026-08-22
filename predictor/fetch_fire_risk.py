"""
산림청 국립산림과학원_산불위험예보정보(시군구 단위)로 북한산이 걸친 지역의
산불위험지수를 받아온다.

사용법:
    export FOREST_FIRE_API_KEY="발급받은 서비스키"
    python fetch_fire_risk.py --out ../frontend/fire_risk.json

주의:
- 요청 파라미터 중 시군구를 지정하는 파라미터명이 공식 문서(미리보기)에 정확히 안 나와있어서,
  가장 흔히 쓰이는 이름인 sigunguCode로 우선 시도한다. 만약 빈 결과나 에러가 나면
  --debug 옵션으로 원본 응답을 보여달라 — 파라미터명만 고치면 될 가능성이 높다.
- 시군구 코드(행정표준코드)는 아래처럼 잘 알려진 값을 썼지만, 이것도 100% 보장은 못 해서
  결과의 sigun/doname 필드가 실제로 우리가 원하는 지역과 일치하는지 확인이 필요하다.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from urllib.parse import unquote

import requests

from http_retry import get_with_retry

BASE_URL = "https://apis.data.go.kr/1400377/forestPointv2/forestPointListSigunguSearchV2"

# 북한산이 걸쳐있는 시군구 (행정표준코드, 5자리) — 서대문구 추가, 실제와 다를 수 있음
BUKHANSAN_SIGUNGU = {
    "11110": "종로구",
    "11305": "강북구",
    "11380": "은평구",
    "11290": "성북구",
    "11410": "서대문구",
    "41281": "고양시 덕양구",
}


def fetch_one(service_key: str, sigungu_code: str, debug: bool = False):
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 100,
        "dataType": "JSON",
        "sigunguCode": sigungu_code,
    }
    resp = get_with_retry(BASE_URL, params, timeout=30)
    try:
        data = resp.json()
    except ValueError:
        if debug:
            print(f"[디버그] JSON 파싱 실패, 원본 응답: {resp.text[:1000]}", file=sys.stderr)
        return []

    if debug:
        print(f"[디버그] {sigungu_code} 원본 응답(앞부분):", json.dumps(data, ensure_ascii=False)[:800], file=sys.stderr)

    body = data.get("response", {}).get("body", data.get("body", {}))
    items = body.get("items", {}) if isinstance(body, dict) else {}
    rows = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(rows, dict):
        rows = [rows]
    return rows or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    service_key = os.environ.get("FOREST_FIRE_API_KEY")
    if not service_key:
        print("환경변수 FOREST_FIRE_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)
    service_key = unquote(service_key)

    all_rows = []
    for code, name in BUKHANSAN_SIGUNGU.items():
        rows = fetch_one(service_key, code, debug=args.debug)
        print(f"[{name}({code})] {len(rows)}건 수신")
        all_rows.extend(rows)

    result = {
        "updatedAt": datetime.now().isoformat(),
        "districts": all_rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.out} (총 {len(all_rows)}건)")


if __name__ == "__main__":
    main()
