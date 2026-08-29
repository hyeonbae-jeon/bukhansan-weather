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
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import requests

from http_retry import get_with_retry

BASE_URL = "https://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnMsg"
# GitHub Actions 실행기(ubuntu-latest)는 시스템 시간대가 UTC라, timezone 정보 없이
# datetime.now()를 쓰면 한국시간(KST, UTC+9)보다 9시간 뒤처진 "지금"이 나온다 — 반드시
# 명시적으로 한국시간 기준으로 계산해야 함 (fetch_current_obs.py 상단 설명 참고).
KST = timezone(timedelta(hours=9))

# 북한산이 걸쳐있는 행정구역 — 이 이름이 특보 통보문 텍스트에 등장하면 우리 지역 특보로 판단.
# 호우·강풍 등은 기상청이 "서북권/동북권" 같은 예보구역 단위로 특보를 내지만, 폭염 등은
# 시/군/구(성북구, 강북구 등) 단위로 내는 경우가 많아서 두 granularity를 모두 넣어둔다.
# (구 단위만 넣으면 예보구역 단위 특보를 놓치고, 반대도 마찬가지라 둘 다 필요함)
BUKHANSAN_DISTRICTS = [
    "서북권", "동북권", "고양",
    "성북구", "강북구", "종로구", "은평구", "서대문구", "덕양구",
]


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

        # "서울(서북권 제외)"처럼 괄호 안에 "OO 제외"라고 일부 지역을 뺀다고
        # 명시하는 경우가 있다. 이걸 그냥 문자열 포함 검사로 훑으면 "제외"라고
        # 적힌 지역명이 오히려 "포함"으로 잘못 잡힌다(예: "서북권"이라는 글자가
        # 괄호 안에 있다는 이유만으로 매칭됨). 괄호 안에서 "제외"로 언급된
        # 지역명들을 먼저 뽑아서 매칭 대상에서 확실히 뺀다.
        excluded = set()
        for m in re.finditer(r"\(([^)]*?)제외\)", areas):
            parts = re.split(r"[,\s·/및]+", m.group(1).strip())
            excluded.update(p for p in parts if p)

        hit_districts = [
            d for d in BUKHANSAN_DISTRICTS
            if d in areas and not any(d in exc for exc in excluded)
        ]

        # "서울"에 특정 권역이 붙어있는지 확인. "서울서남권"처럼 바로 붙기도 하고
        # "서울(서울동남권)"처럼 괄호 안에 들어있기도 해서 폭넓게 잡는다. 여기서는
        # 어떤 권역인지는 안 가리고 "권역이 붙어있는지"만 본다 — 서북권/동북권은
        # 위 BUKHANSAN_DISTRICTS 루프에서 이미 별도로 잡히고, 그 외(서남권/동남권
        # 등 북한산과 무관한 권역)라면 아래 catch-all에서 "그냥 서울"로 잘못
        # 넓게 잡히면 안 되기 때문이다.
        has_seoul_qualifier = bool(re.search(r"서울\s*\(?\s*(?:서울)?\s*[가-힣]{2}권", areas))

        if re.search(r"서울\s*\([^)]*제외\)", areas):
            # "서울(서북권 제외)"처럼 서울 하위 권역 중 하나만 뺀다고 명시된
            # 경우, 실제 특보 문구엔 남은 권역(예: "동북권")이라는 글자가
            # 아예 나오지 않기 때문에 위 단순 포함 검사로는 못 잡는다. 우리가
            # 추적하는 서울 하위 권역(서북권/동북권) 중 제외되지 않은 나머지를
            # 유추해서 넣어준다.
            for r in ("서북권", "동북권"):
                if not any(r in exc for exc in excluded) and r not in hit_districts:
                    hit_districts.append(r)
        elif not has_seoul_qualifier and "서울" in areas:
            # 세분화 없이 그냥 "서울"로만 특보를 내는 경우도 있다(예: 한파,
            # 폭염 등). "서울" 뒤에 어떤 권역이든 붙어있으면(서북권/동북권 포함)
            # 여기서는 건드리지 않는다 — 관련 있는 권역은 이미 위 districts
            # 루프가 잡았고, 관련 없는 권역(서남권/동남권 등)이면 애초에 북한산과
            # 무관하므로 "그냥 서울"로 넓혀서 잡으면 안 된다.
            hit_districts.append("서울")

        if hit_districts:
            matched.append({
                "type": warn_type,
                "matchedDistricts": hit_districts,
                "rawAreaText": areas[:200],  # 너무 길면 잘라서 저장
            })
    return matched


def fetch(service_key: str, debug: bool = False):
    now = datetime.now(KST)
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "stnId": "108",
        "tmfc1": (now - timedelta(days=2)).strftime("%Y%m%d%H%M"),
        "tmfc2": now.strftime("%Y%m%d%H%M"),
    }
    resp = get_with_retry(BASE_URL, params, timeout=30)
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
        result = {"updatedAt": datetime.now(KST).isoformat(), "warnings": []}
    else:
        latest = rows[0]  # 가장 최근 통보문 하나만 사용
        matched = parse_warning_text(latest.get("t6", ""))
        result = {
            "updatedAt": datetime.now(KST).isoformat(),
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
