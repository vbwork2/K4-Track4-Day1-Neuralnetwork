"""(Dành cho giảng viên) Tạo data/split_metadata.csv, tức phép chia train/eval cố định.

Sinh viên KHÔNG cần chạy file này. Họ chỉ chạy scripts/split_data.py.
Phép chia: phân tầng (stratified) theo nhãn, 80% train / 20% eval, seed 42.

    python scripts/make_split_metadata.py
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/covtype.csv.gz")
    ap.add_argument("--out", default="data/split_metadata.csv")
    ap.add_argument("--eval_fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y = pd.read_csv(args.data, usecols=["Cover_Type"])["Cover_Type"].to_numpy()
    row_id = range(len(y))
    train_ids, eval_ids = train_test_split(
        list(row_id), test_size=args.eval_fraction, stratify=y, random_state=args.seed
    )
    split = pd.Series("train", index=row_id, name="split")
    split.loc[eval_ids] = "eval"
    out = pd.DataFrame({"row_id": list(row_id), "split": split.to_numpy()})
    out.to_csv(args.out, index=False)
    print(out["split"].value_counts().to_dict(), "->", args.out)


if __name__ == "__main__":
    main()
