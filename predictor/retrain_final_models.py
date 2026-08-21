"""
실제 서비스에 쓸 최종 모델을 '가진 데이터 100%'로 재학습한다.
(train_model_b_v2.py는 정확도 검증을 위해 15%를 홀드아웃으로 뺐지만,
 배포용은 데이터를 아낄 이유가 없으므로 전부 학습에 사용한다.
 단, LightGBM 조기종료를 위한 검증셋은 마지막 8%를 임시로 떼어 쓰고,
 최적 라운드를 찾은 뒤에는 그 라운드 수만큼 전체 데이터로 다시 학습한다.)

사용법:
    python retrain_final_models.py --table ../data/weather/training_table.csv --out-dir ../data/model_final
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

WEATHER_FEATURES = ["기온", "기준습도", "강수량", "풍속", "전운량"]
TIME_FEATURES = ["시간대", "월", "요일"]
LOC_FEATURES = ["GNSS-경도", "GNSS-위도"]


def add_wind_features(df):
    df = df.copy()
    rad = np.deg2rad(df["풍향"].fillna(0))
    df["풍향_sin"] = np.sin(rad)
    df["풍향_cos"] = np.cos(rad)
    return df


def compute_offset_table(df, target, ref_col):
    tmp = df.copy()
    tmp["시간대구분"] = tmp["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
    tmp["편차"] = tmp[target] - tmp[ref_col]
    return tmp.groupby(["국가지점번호", "시간대구분"])["편차"].mean()


def apply_offset(df, offset_map):
    df = df.copy()
    df["시간대구분"] = df["시간대"].apply(lambda h: "오전" if h < 12 else "오후")
    offset = df.set_index(["국가지점번호", "시간대구분"]).index.map(offset_map)
    return pd.Series(offset, index=df.index).fillna(0.0)


def train_final(df, target, ref_col, feat_cols, out_model_path):
    offset_map = compute_offset_table(df, target, ref_col)
    offset = apply_offset(df, offset_map)
    residual = df[target].values - (df[ref_col].values + offset.values)

    X = df[feat_cols].copy()
    X["국가지점번호"] = df["국가지점번호"].astype("category")

    # 1단계: 최적 라운드 수를 찾기 위해 마지막 8%로 조기종료
    cut = int(len(X) * 0.92)
    X_tr, X_val = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_val = residual[:cut], residual[cut:]

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
    probe = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_iter = probe.best_iteration

    # 2단계: 찾은 라운드 수만큼 전체 데이터(100%)로 다시 학습
    full_set = lgb.Dataset(X, label=residual, categorical_feature=["국가지점번호"])
    final_model = lgb.train(params, full_set, num_boost_round=best_iter)
    final_model.save_model(out_model_path)

    return offset_map, best_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    df["일시"] = pd.to_datetime(df["일시"])
    df = df.sort_values("일시").reset_index(drop=True)
    df = add_wind_features(df)

    feat_cols = [c for c in WEATHER_FEATURES + TIME_FEATURES + LOC_FEATURES if c in df.columns]
    feat_cols += ["풍향_sin", "풍향_cos"]
    df = df.dropna(subset=feat_cols + ["온도", "지점습도", "국가지점번호"])

    os.makedirs(args.out_dir, exist_ok=True)

    all_offsets = {}
    for target, ref_col, label in [("온도", "기온", "temp"), ("지점습도", "기준습도", "humidity")]:
        offset_map, best_iter = train_final(
            df, target, ref_col, feat_cols,
            os.path.join(args.out_dir, f"model_final_{label}.txt"),
        )
        print(f"[{target}] 최종모델 학습 완료 (라운드수={best_iter})")
        all_offsets[label] = {
            f"{p}|{t}": round(float(v), 3) for (p, t), v in offset_map.items()
        }

    with open(os.path.join(args.out_dir, "offsets.json"), "w", encoding="utf-8") as f:
        json.dump(all_offsets, f, ensure_ascii=False, indent=2)

    # 지점 메타정보(좌표, 이름)도 predictor에서 쓰기 위해 같이 저장
    points = (
        df[["국가지점번호", "다목적위치표지판번호", "GNSS-경도", "GNSS-위도"]]
        .drop_duplicates("국가지점번호")
        .to_dict(orient="records")
    )
    with open(os.path.join(args.out_dir, "station_meta.json"), "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)

    print(f"최종 모델/오프셋/지점정보 저장 완료: {args.out_dir}")


if __name__ == "__main__":
    main()
