"""
공공데이터포털 API 호출용 공통 재시도 헬퍼.

GitHub Actions 서버에서 가끔 apis.data.go.kr로의 연결이 일시적으로 타임아웃 나는
경우가 있어서(ConnectTimeout), 실패해도 몇 번 더 시도해보고 그래도 안 되면 그때
에러를 던지도록 감싼 것.
"""
import time
import requests


def get_with_retry(url, params, timeout=30, max_retries=5, backoff_seconds=5):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if not resp.ok:
                # 4xx/5xx는 재시도해도 안 바뀔 가능성이 높아서 재시도는 안 하되,
                # 서버가 실제로 뭐라고 답했는지(원인 파악용) 반드시 출력하고 던진다.
                print(f"[오류] HTTP {resp.status_code} 응답 본문: {resp.text[:1000]}")
                resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            last_exc = e
            print(f"[재시도 {attempt}/{max_retries}] 연결 실패: {e}")
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)  # 5초, 10초, ... 점점 늘려가며 재시도
    raise last_exc
