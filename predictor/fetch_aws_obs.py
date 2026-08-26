"""
기상청 API허브(apihub.kma.go.kr)의 방재기상관측(AWS) 매분자료로, 북한산 주변 5개
실제 관측소의 "지금 실측값"을 받아온다.

왜 필요한가:
지금까지 쓰던 getUltraSrtNcst(초단기실황)는 5km 격자 대표값이라 국지적인 소나기 등을
놓치기 쉬웠다. 이 API는 실제 관측소(AWS)에서 매분 관측한 원본값이라 훨씬 정확하고,
특히 "북한산"이라는 이름의 관측소가 실제로 있어서 산 위 상황을 직접 볼 수 있다.

주의: 공공데이터포털(data.go.kr)의 KMA_API_KEY와는 완전히 다른 시스템(기상자료개방포털
/API허브, data.kma.go.kr 또는 apihub.kma.go.kr)에서 별도로 발급받은 키가 필요하다.
환경변수 KMA_HUB_AUTH_KEY로 넘겨준다.

사용법:
    export KMA_HUB_AUTH_KEY="발급받은 인증키"
    python fetch_aws_obs.py --out ../data/weather/aws_obs.json

응답 형식(AWS 매분자료, disp=1 콤마구분, help=0):
    #START7777
    #  YYMMDDHHMI,STN,WD1,WS1,WDS,WSS,WD10,WS10,TA,RE,RN-15m,RN-60m,RN-12H,RN-DAY,HM,PA,PS,TD
       202601011200,420,...,  -5.2,  0,   0.0,   0.0, ...,   65, 1013.2, 1015.0,  -10.5
    #7777END
자료설명: TA=기온(℃), HM=습도(%), RN-60m=최근1시간 강수량(mm), RE=강수유무(0/1)

※ help=1로 요청하면 각 변수 설명이 여러 줄에 걸쳐 나오는데("#  WD1    : ...",
"#  WS1    : ..." 처럼 한 줄에 변수 하나씩), 그러면 "컬럼명이 한 줄에 다 모여있는
헤더 줄"이 아예 없어서 파싱이 실패한다. help=0으로 바꿔서 이 설명 블록 없이 헤더
줄 하나 + 데이터 줄만 오도록 함.
"""
import argparse
import json
import os
import sys

from http_retry import get_with_retry

BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"

# 북한산 주변 AWS 관측소 — 사용자가 기상청 API허브에서 직접 확인해 준 지점번호.
# ⚠️ 420(북한산)은 실제 조회 결과 "해당 AWS ID는 지점목록에 없습니다" 응답을 받음 —
# 유효한 지점번호가 아닌 것으로 확인됨. 나머지 4개(424/416/414/540)는 정상 응답.
# 북한산 지점의 올바른 번호를 다시 확인하기 전까지는 이 4개만 씀.
BUKHANSAN_AWS_STATIONS = {
    "424": "강북",
    "416": "은평",
    "414": "성북",
    "540": "고양",
}


def parse_aws_response(text: str) -> dict | None:
    """콤마구분(disp=1) 텍스트 응답에서 헤더 줄(컬럼명)을 찾아 그 이름으로 값을
    매핑한다. 위치가 아니라 이름으로 찾기 때문에, 실제 컬럼 순서가 문서와 살짝
    달라도 안전하게 파싱된다. 여러 줄(시간 범위)이 오면 가장 마지막(=가장 최근)
    관측값을 쓴다."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_cols = None
    last_row = None
    for ln in lines:
        if ln.startswith("#") and "TA" in ln and "STN" in ln:
            header_cols = [c.strip().lstrip("#").strip() for c in ln.split(",")]
            continue
        if ln.startswith("#"):
            continue
        if header_cols is None:
            continue
        values = [v.strip() for v in ln.split(",")]
        if len(values) != len(header_cols):
            continue
        last_row = dict(zip(header_cols, values))
    return last_row


def fetch_station(auth_key: str, stn: str) -> dict | None:
    params = {"tm2": "", "stn": stn, "disp": 1, "help": 0, "authKey": auth_key}
    resp = get_with_retry(BASE_URL, params, timeout=30)
    # 이 API는 EUC-KR로 응답을 주는데, requests가 기본으로 다른 인코딩으로 잘못
    # 추측해서 한글 설명이 깨져 나오는 문제가 있었음 — 명시적으로 지정해서 해결.
    resp.encoding = "euc-kr"
    row = parse_aws_response(resp.text)
    if row is None:
        print(f"  경고: 지점 {stn} 파싱 실패 — 원본 응답 앞부분: {resp.text[:300]!r}", file=sys.stderr)
        return None

    def to_float(key):
        v = row.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "obsTime": row.get("YYMMDDHHMI"),
        "temp": to_float("TA"),
        "humidity": to_float("HM"),
        "rain1h": to_float("RN-60m"),
        "rainNow": to_float("RE"),  # 강수유무(0/1) — 참고용
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    auth_key = os.environ.get("KMA_HUB_AUTH_KEY")
    if not auth_key:
        print("환경변수 KMA_HUB_AUTH_KEY가 설정되어 있지 않습니다 (data.kma.go.kr/apihub.kma.go.kr에서 "
              "별도 발급 — 공공데이터포털 KMA_API_KEY와는 다른 키).", file=sys.stderr)
        sys.exit(1)

    result = {}
    for stn, name in BUKHANSAN_AWS_STATIONS.items():
        print(f"[fetch] {stn} ({name})")
        try:
            obs = fetch_station(auth_key, stn)
            if obs:
                obs["name"] = name
                result[stn] = obs
        except Exception as e:
            print(f"  경고: {stn}({name}) 조회 실패 - {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.out} ({len(result)}개 관측소)")


if __name__ == "__main__":
    main()
