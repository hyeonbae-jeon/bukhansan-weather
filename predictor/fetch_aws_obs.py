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

응답 형식(AWS 매분자료, disp=1 콤마구분):
    #START7777
    #  YYMMDDHHMI,STN,WD1,WS1,WDS,WSS,WD10,WS10,TA,RE,RN-15m,RN-60m,RN-12H,RN-DAY,HM,PA,PS,TD
       202601011200,420,...,  -5.2,  0,   0.0,   0.0, ...,   65, 1013.2, 1015.0,  -10.5
    #7777END
자료설명: TA=기온(℃), HM=습도(%), RN-60m=최근1시간 강수량(mm), RE=강수유무(0/1)
"""
import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from http_retry import get_with_retry

BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
KST = timezone(timedelta(hours=9))

# 북한산 주변 AWS 관측소. 예전엔 "북한산"(420)이 있는 줄 알았는데, 실제로
# 요청해보니 "해당 AWS ID는 지점목록에 없습니다"라는 응답이 왔고, 이번에
# 기상청 API허브 공식 지점정보(inf=AWS)로 확인해봐도 420은 목록에 없는 번호였다
# — 화면 캡처로 확인했던 번호가 다른 종류의 지점 코드였던 것으로 보임. 그래서
# 뺐고, 대신 공식 지점정보로 좌표까지 다시 맞춘 4개 관측소에 공원 북동쪽(도봉산/
# 사패산 방향)을 봐줄 "도봉"(406)을 새로 추가했다.
BUKHANSAN_AWS_STATIONS = {
    "424": "강북",
    "416": "은평",
    "414": "성북",
    "540": "고양",
    "406": "도봉",
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
    # 기상청 API허브 공식 문서: tm2는 "없으면 현재시간"으로 기본 처리된다고 되어
    # 있지만, 그건 파라미터 자체가 빠졌을 때 얘기고 빈 문자열("")을 값으로 보내면
    # 서버가 이를 잘못된 값으로 보고 실제 데이터 대신 도움말(필드 설명)만 돌려준다
    # (AWS 수집이 계속 실패하던 근본 원인이 이거였음). 그래서 tm2를 실제 KST
    # 시각으로 명시하고, 매분자료가 아직 안 올라왔을 시점 대비 tm1을 10분 전으로
    # 잡아 범위로 요청한다 — parse_aws_response가 그중 가장 마지막(최신) 행을 쓴다.
    now_kst = datetime.now(KST)
    tm2 = now_kst.strftime("%Y%m%d%H%M")
    tm1 = (now_kst - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    params = {"tm1": tm1, "tm2": tm2, "stn": stn, "disp": 1, "help": 1, "authKey": auth_key}
    resp = get_with_retry(BASE_URL, params, timeout=30)
    resp.encoding = "euc-kr"  # 이 API는 EUC-KR로 응답 — 지정 안 하면 한글 설명이 깨져 보임
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

    # 실패했을 때 되돌아갈 "직전 값" — 기존 --out 파일이 있으면 읽어둔다
    previous = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as f:
                previous = json.load(f)
        except Exception as e:
            print(f"  경고: 기존 파일을 못 읽어서 실패 시 직전 값 유지를 못 함 - {e}", file=sys.stderr)

    # 관측소 5개를 병렬로 조회 (fetch_forecast.py 등과 동일한 이유 — 순차 호출 시
    # 한 관측소의 재시도 실패가 그대로 전체 소요시간에 더해짐. 실제로 이 스크립트가
    # 아직 순차 방식이던 때, apis 쪽 장애 상황에서 혼자 17분을 잡아먹은 적이 있었음)
    def _fetch_one(item):
        stn, name = item
        try:
            return stn, name, fetch_station(auth_key, stn), None
        except Exception as e:
            return stn, name, None, e

    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_fetch_one, item) for item in BUKHANSAN_AWS_STATIONS.items()]
        for future in concurrent.futures.as_completed(futures):
            stn, name, obs, err = future.result()
            if err is not None:
                if stn in previous:
                    print(f"  경고: {stn}({name}) 조회 실패(직전 값 유지) - {err}", file=sys.stderr)
                    result[stn] = previous[stn]
                else:
                    print(f"  경고: {stn}({name}) 조회 실패(직전 값도 없음) - {err}", file=sys.stderr)
                continue
            if obs is None:
                # fetch_station이 파싱 실패 등으로 None을 반환한 경우도 마찬가지로
                # 직전 값을 유지한다 (예외는 아니지만 실질적으로 이번 회차 실패임)
                if stn in previous:
                    result[stn] = previous[stn]
                continue
            print(f"[fetch] {stn} ({name}) 완료")
            obs["name"] = name
            result[stn] = obs

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.out} ({len(result)}개 관측소)")


if __name__ == "__main__":
    main()
