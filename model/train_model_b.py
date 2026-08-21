"""
모델 B: 기상 조건에 따라 지점별 편차가 달라지는 것을 학습하는 조건부 ML 모델.

입력 특성: 기준 관측소 기온/습도/강수량/풍속/풍향/전운량, 시간대, 월(계절), 지점 위경도
목표: 지점 실측 온도 / 습도 (편차가 아니라 절대값을 직접 예측 — 기준 관측소 값이 특성으로
      이미 들어가 있어서 모델이 알아서 "기준값 + 조건부 보정"을 학습하게 된다)

기간 앞부분 85%로 학습, 뒤 15%(시간순 최신 구간)를 홀드아웃으로 남겨 정직하게 평가하고,
같은 홀드아웃에서 모델 A(정적 오프셋)와 정확도를 비교해서 리포트를 남긴다.

사용법:
    python train_model_b.py --table ../data/weather/training_table.csv \
        --out-dir ../data/model_b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "기온", "기준습도", "강수량", "풍속", "풍향", "전운량",
    "시간대", "월", "요일", "GNSS-경도", "GNSS-위도",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 풍향은 각도(0~360)라 그대로 쓰면 359도와 1도가 멀게 학습됨 -> sin/cos로 변환
    rad = np.deg2rad(df["풍향"].fillna(0))
    df["풍향_sin"] = np.sin(rad)
    df["풍향_cos"] = np.cos(rad)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df.sort_values("일시").reset_index(drop=True)
    df = build_features(df)

    feat_cols = [c for c in FEATURES if c != "풍향"] + ["풍향_sin", "풍향_cos"]
    # API 응답에 없는 컬럼(예: 전운량이 결측/미제공)은 조용히 제외 — 있는 특성만으로 학습
    available = [c for c in feat_cols if c in df.columns]
    dropped = [c for c in feat_cols if c not in df.columns]
    if dropped:
        print(f"[안내] 데이터에 없는 특성이라 제외함: {dropped}")
    feat_cols = available
    df = df.dropna(subset=feat_cols + ["온도", "지점습도"])

    split_idx = int(len(df) * (1 - args.holdout_frac))
    split_time = df.iloc[split_idx]["일시"]
    train_df = df[df["일시"] < split_time]
    test_df = df[df["일시"] >= split_time]
    print(f"학습 {len(train_df)}행 / 홀드아웃 {len(test_df)}행 (기준시각: {split_time})")

    os.makedirs(args.out_dir, exist_ok=True)

    models = {}
    report = {"홀드아웃_기준시각": str(split_time), "지표": {}}

    for target, label in [("온도", "temp"), ("지점습도", "humidity")]:
        train_set = lgb.Dataset(train_df[feat_cols], label=train_df[target])
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 30,
            "verbose": -1,
        }
        model = lgb.train(params, train_set, num_boost_round=300)
        model.save_model(os.path.join(args.out_dir, f"model_b_{label}.txt"))
        models[target] = model

        pred = model.predict(test_df[feat_cols])
        mae_b = mean_absolute_error(test_df[target], pred)

        # 비교용 모델 A(단순 평균): 학습 구간에서만 지점×시간대 평균 편차를 계산해서
        # 홀드아웃에 적용 (미래 데이터 누수 방지)
        ref_col = "기온" if target == "온도" else "기준습도"
        train_df2 = train_df.copy()
        train_df2["시간대구분"] = train_df2["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
        offset_map = (
            train_df2.assign(편차=train_df2[target] - train_df2[ref_col])
            .groupby(["국가지점번호", "시간대구분"])["편차"].mean()
        )

        test_df2 = test_df.copy()
        test_df2["시간대구분"] = test_df2["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
        test_df2["_offset"] = test_df2.set_index(["국가지점번호", "시간대구분"]).index.map(offset_map)
        pred_a = test_df2[ref_col] + test_df2["_offset"].fillna(0)
        mae_a = mean_absolute_error(test_df2[target], pred_a)

        report["지표"][target] = {
            "모델A_MAE": round(float(mae_a), 3),
            "모델B_MAE": round(float(mae_b), 3),
            "개선폭": round(float(mae_a - mae_b), 3),
        }
        print(f"[{target}] 모델A MAE={mae_a:.3f} / 모델B MAE={mae_b:.3f}")

    with open(os.path.join(args.out_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"모델 및 평가 리포트 저장 완료: {args.out_dir}")


if __name__ == "__main__":
    main()
