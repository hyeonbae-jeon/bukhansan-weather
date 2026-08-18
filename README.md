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
python train_model_b.py --table ../data/weather/training_table.csv --out-dir ../data/model_b
```
`train_model_b.py`를 실행하면 마지막에 **모델 A vs 모델 B 정확도 비교(MAE)**가
`data/model_b/evaluation_report.json`에 저장됩니다. 이 결과를 보고 실제 사이트에서
두 모델을 어떻게 보여줄지(둘 다 보여줄지, B가 확실히 낫다면 B만 쓸지) 다음 단계에서 정하면 돼요.

## 코드 검증 상태
`fetch_kma_hourly.py`는 실제 API 키가 있어야 호출되기 때문에 이 환경(외부 네트워크 제한)에서는
직접 실행해보지 못했습니다. 대신 `merge_sensor_weather.py`, `train_model_a.py`, `train_model_b.py`는
더미(가상) 기상 데이터로 전체 파이프라인이 에러 없이 끝까지 도는 것까지 확인했습니다.
1단계(기상청 API 호출)만 직접 실행해서 정상적으로 CSV가 떨어지는지 확인해주세요 — 만약 API 응답
구조나 컬럼명이 예상과 다르면(공공데이터포털 API는 가끔 스펙이 바뀌어요) 알려주시면 바로 고칠게요.

## 출처 표시 (필수)
공공누리 4유형(출처표시+상업적 이용금지+변경금지) 조건이라, 사이트 하단에 아래 표기가 필요해요:
- 자료출처: 서울특별시 (서울 열린데이터광장)
- 자료출처: 기상청
