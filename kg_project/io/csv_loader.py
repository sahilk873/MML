from pathlib import Path

import pandas as pd


def read_label_csvs(src_dir: Path, pos_csv: str, neg_csv: str) -> pd.DataFrame:
    pos_path = src_dir / pos_csv
    neg_path = src_dir / neg_csv
    pos_df = pd.read_csv(pos_path, dtype=str)
    neg_df = pd.read_csv(neg_path, dtype=str)
    pos_df["label"] = 1
    neg_df["label"] = 0
    combined = pd.concat([pos_df, neg_df], ignore_index=True, sort=False)
    return combined


def save_labels_clean(df: pd.DataFrame, out_dir: Path):
    path = out_dir / "labels_clean.parquet"
    df.to_parquet(path, index=False)
