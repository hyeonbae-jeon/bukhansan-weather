"""
모델 B (개선판): 모델 A(지점×시간대 정적 오프셋)를 베이스라인으로 깔고,
그 위에서 "날씨 조건에 따라 오프셋이 어떻게 달라지는지"의 잔차(residual)만 학습한다.

이렇게 하면:
- 조건부 신호가 약해도 최소한 모델 A보다 나빠지지 않는다 (잔차가 0에 가까우면 그냥 모델 A와 같아짐)
- 조건부 신호가 있으면 그만큼 정확도가 더 올라간다

추가 개선:
- 지점(국가지점번호)을 범주형 특성으로 직접 투입 (LightGBM 네이티브 categorical)
- 정규화(L1/L2, feature/bagging fraction) + 조기종료(early stopping)로 과적합 방지

사용법:
    python train_model_b.py --table ../data/weather/training_table.csv --out-dir ../data/model_b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error


WEATHER_FEATURES = ["기온", "기준습도", "강수량", "풍속", "전운량"]
TIME_FEATURES = ["시간대", "월", "요일"]
LOC_FEATURES = ["GNSS-경도", "GNSS-위도"]


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rad = np.deg2rad(df["풍향"].fillna(0))
    df["풍향_sin"] = np.sin(rad)
    df["풍향_cos"] = np.cos(rad)
    return df


def compute_offset_table(train_df: pd.DataFrame, target: str, ref_col: str) -> pd.Series:
    """학습 구간에서만 지점×시간대구분 평균 편차를 계산 (모델 A와 동일 로직, 미래 데이터 누수 없음)"""
    tmp = train_df.copy()
    tmp["시간대구분"] = tmp["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
    tmp["편차"] = tmp[target] - tmp[ref_col]
    return tmp.groupby(["국가지점번호", "시간대구분"])["편차"].mean()


def apply_offset(df: pd.DataFrame, offset_map: pd.Series) -> pd.Series:
    df = df.copy()
    df["시간대구분"] = df["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
    offset = df.set_index(["국가지점번호", "시간대구분"]).index.map(offset_map)
    return pd.Series(offset, index=df.index).fillna(0.0)


def train_one_target(train_df, test_df, target, ref_col, feat_cols, out_path):
    offset_map = compute_offset_table(train_df, target, ref_col)

    train_offset = apply_offset(train_df, offset_map)
    test_offset = apply_offset(test_df, offset_map)

    # 잔차 = 실제값 - (기준관측값 + 정적오프셋) = 모델A가 못 잡아낸 나머지
    train_residual = train_df[target].values - (train_df[ref_col].values + train_offset.values)
    test_residual = test_df[target].values - (test_df[ref_col].values + test_offset.values)

    X_train = train_df[feat_cols].copy()
    X_train["국가지점번호"] = train_df["국가지점번호"].astype("category")
    X_test = test_df[feat_cols].copy()
    X_test["국가지점번호"] = pd.Categorical(
        test_df["국가지점번호"], categories=X_train["국가지점번호"].cat.categories
    )

    # 학습 구간 내에서 마지막 10%를 조기종료용 검증셋으로 분리 (시간순 유지)
    val_cut = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr, y_val = train_residual[:val_cut], train_residual[val_cut:]

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=["국가지점번호"])
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=["국가지점번호"], reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    model.save_model(out_path)

    pred_residual = model.predict(X_test, num_iteration=model.best_iteration)
    pred_final = test_df[ref_col].values + test_offset.values + pred_residual

    mae_b = mean_absolute_error(test_df[target], pred_final)
    mae_a = mean_absolute_error(test_df[target], test_df[ref_col].values + test_offset.values)
    return mae_a, mae_b, model.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df.sort_values("일시").reset_index(drop=True)
    df = add_wind_features(df)

    feat_cols = [c for c in WEATHER_FEATURES + TIME_FEATURES + LOC_FEATURES if c in df.columns]
    feat_cols += ["풍향_sin", "풍향_cos"]
    dropped = [c for c in WEATHER_FEATURES if c not in df.columns]
    if dropped:
        print(f"[안내] 데이터에 없는 특성이라 제외함: {dropped}")

    df = df.dropna(subset=feat_cols + ["온도", "지점습도", "국가지점번호"])

    split_idx = int(len(df) * (1 - args.holdout_frac))
    split_time = df.iloc[split_idx]["일시"]
    train_df = df[df["일시"] < split_time]
    test_df = df[df["일시"] >= split_time]
    print(f"학습 {len(train_df)}행 / 홀드아웃 {len(test_df)}행 (기준시각: {split_time})")

    os.makedirs(args.out_dir, exist_ok=True)
    report = {"홀드아웃_기준시각": str(split_time), "지표": {}}

    for target, ref_col, label in [("온도", "기온", "temp"), ("지점습도", "기준습도", "humidity")]:
        mae_a, mae_b, best_iter = train_one_target(
            train_df, test_df, target, ref_col, feat_cols,
            os.path.join(args.out_dir, f"model_b_{label}.txt"),
        )
        report["지표"][target] = {
            "모델A_MAE": round(float(mae_a), 3),
            "모델B_MAE": round(float(mae_b), 3),
            "개선폭": round(float(mae_a - mae_b), 3),
            "최적_부스팅라운드": int(best_iter),
        }
        print(f"[{target}] 모델A MAE={mae_a:.3f} / 모델B MAE={mae_b:.3f} (best_iter={best_iter})")

    with open(os.path.join(args.out_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"모델 및 평가 리포트 저장 완료: {args.out_dir}")


if __name__ == "__main__":
    main()
