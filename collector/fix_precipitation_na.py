"""
이미 fetch_kma_hourly.py로 받아둔 CSV의 강수량 결측치를 0으로 채우는 1회성 수정 스크립트.
(fetch_kma_hourly.py 자체는 이미 고쳐져 있어서, 이 파일은 '이전에 받아둔 파일'만 고치면 될 때 씀)

사용법:
    python fix_precipitation_na.py --in ../data/weather/asos_108.csv
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
args = ap.parse_args()

df = pd.read_csv(args.inp)
before = df["강수량"].isna().sum()
df["강수량"] = df["강수량"].fillna(0.0)
df.to_csv(args.inp, index=False)
print(f"강수량 결측 {before}건을 0으로 채우고 덮어썼습니다: {args.inp}")
