"""
모델 A: 지점별 정적 편차(고정 오프셋) 계산.

기상청 예보값에 "이 지점은 평균적으로 기준 관측소보다 온도 -X도, 습도 +Y%"를
그대로 더해서 예측하는 가장 단순한 방식.
시간대(오전/오후)별로 편차가 다를 수 있어 시간대별로 나눠서 계산한다.

사용법:
    python train_model_a.py --table ../data/weather/training_table.csv --out ../data/model_a_offsets.json
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.table)

    # 시간대를 '오전'(0~11시) / '오후'(12~23시)로 단순화 — 센서가 하루 2회(06시경/18시경)만 찍혀서
    # 세밀한 시간대 구분은 의미가 없다.
    df["시간대구분"] = df["시간대"].apply(lambda h: "오전" if h < 12 else "오후")

    grouped = (
        df.groupby(["국가지점번호", "다목적위치표지판번호", "시간대구분"])
        .agg(
            온도편차_평균=("온도편차", "mean"),
            온도편차_표준편차=("온도편차", "std"),
            습도편차_평균=("습도편차", "mean"),
            습도편차_표준편차=("습도편차", "std"),
            표본수=("온도편차", "count"),
            경도=("GNSS-경도", "first"),
            위도=("GNSS-위도", "first"),
        )
        .reset_index()
    )

    result = {}
    for _, row in grouped.iterrows():
        key = row["국가지점번호"]
        result.setdefault(key, {
            "지점명": row["다목적위치표지판번호"],
            "경도": row["경도"],
            "위도": row["위도"],
            "시간대별편차": {},
        })
        result[key]["시간대별편차"][row["시간대구분"]] = {
            "온도편차": round(row["온도편차_평균"], 2),
            "온도편차_표준편차": round(row["온도편차_표준편차"], 2) if pd.notna(row["온도편차_표준편차"]) else None,
            "습도편차": round(row["습도편차_평균"], 2),
            "습도편차_표준편차": round(row["습도편차_표준편차"], 2) if pd.notna(row["습도편차_표준편차"]) else None,
            "표본수": int(row["표본수"]),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {args.out} ({len(result)}개 지점)")


if __name__ == "__main__":
    main()
