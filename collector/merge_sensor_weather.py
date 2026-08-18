"""
센서 데이터(지점별 실측 온습도)와 기상청 ASOS 시간자료(기준 관측소)를
가장 가까운 시각 기준으로 매칭해서, 학습용 테이블을 만든다.

사용법:
    python merge_sensor_weather.py \
        --sensor ../data/sensor/sensor_merged.csv \
        --weather ../data/weather/asos_108.csv \
        --out ../data/weather/training_table.csv
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-min", type=int, default=90,
                     help="센서 관측 시각과 기상청 관측 시각의 최대 허용 차이(분)")
    args = ap.parse_args()

    sensor = pd.read_csv(args.sensor)
    sensor["일시"] = pd.to_datetime(sensor["일시"])

    weather = pd.read_csv(args.weather)
    weather["관측시각"] = pd.to_datetime(weather["관측시각"])
    weather = weather.sort_values("관측시각")

    sensor = sensor.sort_values("일시")

    merged = pd.merge_asof(
        sensor,
        weather,
        left_on="일시",
        right_on="관측시각",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=args.max_gap_min),
    )

    humidity_col = "습도_y" if "습도_y" in merged.columns else "습도"
    before = len(merged)
    merged = merged.dropna(subset=["기온", humidity_col])
    after = len(merged)
    print(f"기상 데이터 매칭 성공: {after}/{before} 행 (허용오차 {args.max_gap_min}분 이내)")

    # 편차(deviation) 컬럼 계산: 지점 실측값 - 기준 관측소값
    # 컬럼명이 sensor/weather 양쪽 다 '습도'라 겹치므로 pandas가 붙이는 _x/_y 접미사로 구분
    if "습도_x" in merged.columns:
        merged["지점습도"] = merged["습도_x"]
        merged["기준습도"] = merged["습도_y"]
    else:
        merged["지점습도"] = merged["습도"]
        merged["기준습도"] = merged["습도"]

    merged["온도편차"] = merged["온도"] - merged["기온"]
    merged["습도편차"] = merged["지점습도"] - merged["기준습도"]

    # 시간 특성
    merged["시간대"] = merged["일시"].dt.hour
    merged["월"] = merged["일시"].dt.month
    merged["요일"] = merged["일시"].dt.dayofweek

    keep = [
        "일시", "국가지점번호", "다목적위치표지판번호", "GNSS-경도", "GNSS-위도",
        "온도", "지점습도", "기온", "기준습도",
        "강수량", "풍속", "풍향", "전운량",
        "온도편차", "습도편차", "시간대", "월", "요일",
    ]
    keep = [c for c in keep if c in merged.columns]
    merged = merged[keep]

    merged.to_csv(args.out, index=False)
    print(f"저장 완료: {args.out} ({len(merged)}행, {merged['다목적위치표지판번호'].nunique()}개 지점)")


if __name__ == "__main__":
    main()
