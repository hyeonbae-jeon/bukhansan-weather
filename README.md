# 북한산 지점별 날씨 예측 사이트 (데이터 파이프라인)

북한산 국가지점번호 IoT 센서 데이터(서울 열린데이터광장) + 기상청 관측/예보 데이터를 결합해
등산로 지점별 날씨를 예측하는 프로젝트의 1단계(데이터 파이프라인) 산출물입니다.

## 지금까지 확보한 것
- `data/sensor/sensor_merged.csv`: 2025-01 ~ 2026-05, 13개월치 센서 데이터 병합본 (119개 지점, 83,664행)
- 기상청 API 인증키 2개 발급 완료 (ASOS 시간자료, 단기예보)

## 폴더 구조
```
collector/
  fetch_kma_hourly.py       # 기상청 ASOS 시간자료 수집 (실행 필요)
  merge_sensor_weather.py   # 센서 데이터 + 기상 데이터 매칭 (실행 필요)
model/
  train_model_a.py          # 모델 A: 지점별 정적 편차 오프셋
  train_model_b.py          # 모델 B: 조건부 ML 보정 모델 (LightGBM) + A/B 비교 리포트
predictor/                  # (다음 단계) 실시간 예보 반영해서 예측 생성
data/
  sensor/sensor_merged.csv  # 이미 있음
  weather/                  # 아래 1단계 실행하면 여기 생성됨
```

## 실행 순서 (로컬 또는 GitHub Actions)

### 1. 기상청 과거 시간자료 수집
```bash
export KMA_API_KEY="발급받은 서비스키(Decoding 키 권장)"
cd collector
python fetch_kma_hourly.py --start 20250101 --end 20260531 --out ../data/weather/asos_108.csv
```
데이터가 13개월치라 API 호출이 꽤 걸릴 수 있어요(90일 단위로 나눠서 호출, 자동 페이지네이션 처리됨).

### 2. 센서 데이터와 매칭
```bash
python merge_sensor_weather.py \
  --sensor ../data/sensor/sensor_merged.csv \
  --weather ../data/weather/asos_108.csv \
  --out ../data/weather/training_table.csv
```

### 3. 모델 학습
```bash
cd ../model
python train_model_a.py --table ../data/weather/training_table.csv --out ../data/model_a_offsets.json
python train_model_b_v2.py --table ../data/weather/training_table.csv --out-dir ../data/model_b
```
`evaluation_report.json`으로 모델 A vs B 정확도(MAE)를 비교할 수 있어요. (현재까지 실측 결과: 온도 1.221°C, 습도 6.745%로 모델 B가 확실히 더 정확함을 확인함)

### 4. 최종 배포용 모델 재학습 (100% 데이터)
검증(3번 단계)은 데이터 15%를 떼어 정확도를 확인하기 위한 것이었고, 실제 서비스에는 가진 데이터를 전부 써서 다시 학습해요.
```bash
cd ../predictor
python retrain_final_models.py --table ../data/weather/training_table.csv --out-dir ../data/model_final
```

### 5. 기상청 단기예보 수집
```bash
export KMA_API_KEY="발급받은 서비스키"
python fetch_forecast.py --sensor ../data/sensor/sensor_merged.csv --out ../data/weather/forecast.csv
```

### 6. 지점별 예측 생성
```bash
python generate_predictions.py \
  --forecast ../data/weather/forecast.csv \
  --model-dir ../data/model_final \
  --sensor ../data/sensor/sensor_merged.csv \
  --extra-points ../data/extra_points.csv \
  --out ../frontend/points_predictions.json
```
`frontend/points_predictions.json`에 실측 모델 지점(120개) + 온습도 센서가 없는 추가 지점
(`data/extra_points.csv`, 75개 — 북한산 전체 187개 지점 중 실측 데이터가 없는 곳들을
주변 실측 지점으로 IDW 보간)까지 합쳐 저장돼요. `--extra-points`는 선택 옵션이라 안 넣으면
기존처럼 실측 모델 지점만 생성돼요.

## 7. 프론트엔드 (카카오맵 2D / 브이월드 3D)
`frontend/index.html` 하나로 된 사이트예요. 열기 전에:

1. `index.html` 상단의 `CONFIG` 객체에 카카오 JS 키와 브이월드 API 키를 넣으세요.
   ```js
   const CONFIG = {
     KAKAO_JS_KEY: "발급받은 카카오 JS 키",
     VWORLD_API_KEY: "발급받은 브이월드 API 키",
     ...
   };
   ```
2. **`file://`로 직접 열면 안 돼요** — fetch()로 JSON/GeoJSON을 불러오는데 브라우저가 로컬 파일 접근을 막아요.
   `frontend` 폴더에서 간단한 로컬 서버를 띄워서 열어주세요:
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   그리고 브라우저에서 `http://localhost:8000` 접속.
3. 카카오 개발자센터에서 이 사이트가 열릴 도메인(로컬 테스트는 `http://localhost:8000`)을 **플랫폼 등록**에 추가해야 지도가 떠요.

### 만든 것
- 카카오맵(2D) ↔ 브이월드(3D) 버튼 전환
- 예보 시각 슬라이더로 전체 120개 지점 마커 색이 온도에 따라 실시간으로 바뀜
- 지점 클릭 시 우측 패널에 상세 예보(온도/습도 전체 시간대) 표시
- 북한산 국립공원 경계, 탐방로(102개 구간) 오버레이 (카카오 2D 지도 기준, on/off 토글 가능)
- 하단에 공공데이터 출처 표기 (공공누리 조건 충족용)

### 검증 상태 / 한계
- **카카오맵(2D) 로직**: 마커, 경계 폴리곤, 탐방로 폴리라인 좌표 변환 로직을 실제 GeoJSON 파일로 구조 검증했고 JS 문법 오류도 없음을 확인했어요. 다만 실제 카카오 SDK 호출은 이 환경 네트워크 제한으로 직접 테스트 못 했어요 — 키 넣고 열어보시고 이상 있으면 알려주세요.
- **브이월드(3D)**: 공식 WebGL 3D API 3.0 샘플 구조를 그대로 따랐지만, VWorld는 API 문서가 자주 바뀌고 이 환경에서 실제 키로 테스트가 불가능해서 `vw.Feature.Point` 등 일부 메서드명이 최신 스펙과 다를 수 있어요. 3D 지도가 뜨는데 마커가 안 보이거나 에러가 나면, 브라우저 개발자도구(F12) 콘솔에 뜨는 에러 메시지를 보여주세요 — 바로 고칠게요. 3D 지도에는 아직 경계/탐방로 오버레이는 넣지 않았어요(VWorld 3D의 GeoJSON 레이어 API가 불확실해서 2D 먼저 확실하게 만들었어요).
- 데이터 파일(`points_predictions.json`, `geo/boundary.geojson`, `geo/trails.geojson`)은 전부 이미 폴더에 들어있어요.

## 출처 표시 (필수)
공공누리 4유형(출처표시+상업적 이용금지+변경금지) 조건이라, 사이트 하단에 아래 표기가 필요해요:
- 자료출처: 서울특별시 (서울 열린데이터광장)
- 자료출처: 기상청
- 자료출처: 국토교통부 (브이월드)
- 지도: 카카오맵

## GitHub로 옮기기 (호스팅 + 예보 자동 갱신)

### 준비: `data/model_final` 폴더를 꼭 채워넣으세요
로컬에서 `predictor/retrain_final_models.py`를 돌려서 나온 결과물
(`model_final_temp.txt`, `model_final_humidity.txt`, `offsets.json`, `station_meta.json`)이
`data/model_final/` 안에 있어야 자동 갱신 워크플로가 돌아가요. 이 zip에는 안 들어있으니 로컬에서
직접 넣고 커밋해주세요.

### 1) 저장소 만들고 푸시
```bash
cd bukhansan-weather
git init
git add .
git commit -m "북한산 날씨 예보 사이트 초기 커밋"
gh repo create bukhansan-weather --public --source=. --push
# gh(GitHub CLI)가 없으면: github.com에서 새 저장소 만들고
# git remote add origin <저장소 URL> && git push -u origin main
```

### 2) 기상청 API 키를 GitHub Secret으로 등록
저장소 > Settings > Secrets and variables > Actions > New repository secret
- 이름: `KMA_API_KEY` / 값: 단기예보·초단기실황 서비스키
- 이름: `KMA_WARN_API_KEY` / 값: 기상특보 조회서비스 서비스키 (공공데이터포털은 계정당 인증키 하나라 KMA_API_KEY와 같은 값 넣으면 돼요)
- 이름: `FOREST_FIRE_API_KEY` / 값: 산림청 산불위험예보정보 서비스키

### 3) GitHub Pages 활성화
저장소 > Settings > Pages > **Source: GitHub Actions** 선택 (브랜치 선택 아님, 반드시 "GitHub Actions"로)

### 4) 카카오/브이월드 키를 실제 배포 도메인에 등록
Pages가 배포되면 `https://<계정명>.github.io/bukhansan-weather/` 같은 주소가 생겨요.
- 카카오 개발자센터: 내 애플리케이션 > 플랫폼 > Web에 이 주소 등록
- 브이월드: 오픈API > 인증키 관리에서 도메인 등록

### 5) `index.html`의 CONFIG에 키 입력하고 커밋
`frontend/index.html` 상단 `CONFIG`에 카카오/브이월드 키를 넣고 커밋·푸시하면
`.github/workflows/deploy-pages.yml`이 자동으로 사이트를 배포해요.

### 이후 자동으로 도는 것
- **`.github/workflows/update-predictions.yml`**: 매시 정각 기상청 예보를 새로 받아서
  `points_predictions.json`을 갱신하고 커밋 → 자동으로 사이트도 재배포됨
- **`.github/workflows/deploy-pages.yml`**: `frontend/` 폴더가 바뀔 때마다 자동 배포

### 참고: apis.data.go.kr 연결이 가끔 실패할 수 있음
GitHub 클라우드 서버에서 공공데이터포털로 연결이 가끔 타임아웃 나는 경우가 있어요(완전히
막힌 건 아니고 간헐적). `http_retry.py`에서 최대 5번까지 자동 재시도하도록 되어있고,
그래도 실패하면 그 시간 갱신만 건너뛰고 다음 정각에 다시 자동으로 시도해요 — 매시간
도는 구조라 한 번 실패해도 크게 문제는 없어요. Actions 탭에서 자꾸 실패한다면 실패한
단계의 로그를 확인해보세요.

### 자동화 안 되는 것 (수동으로 해야 함)
- 북한산 센서 CSV(서울 열린데이터광장)는 다운로드 버튼이 자바스크립트라 자동 수집이 안 돼요.
  새 달치가 나오면 직접 받아서 `data/sensor/`에 추가하고, `merge_sensor_weather.py` →
  `retrain_final_models.py`를 다시 돌려서 모델을 갱신해야 해요. (매달 할 필요는 없고,
  몇 달에 한 번 정도면 충분해요.)
