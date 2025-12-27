from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_random_splits(df: pd.DataFrame, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=seed)
    train_idx, rest_idx = next(splitter.split(df, df["label"]))
    train = df.iloc[train_idx].reset_index(drop=True)
    rest = df.iloc[rest_idx].reset_index(drop=True)
    splitter2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=seed + 1)
    val_idx, test_idx = next(splitter2.split(rest, rest["label"]))
    val = rest.iloc[val_idx].reset_index(drop=True)
    test = rest.iloc[test_idx].reset_index(drop=True)
    return train, val, test


def _safe_split_pair_keys(pair_keys: list[str], seed: int):
    rng = np.random.default_rng(seed)
    rng.shuffle(pair_keys)
    n = len(pair_keys)
    if n == 0:
        return [], [], []
    if n == 1:
        return pair_keys, [], []
    if n == 2:
        return [pair_keys[0]], [], [pair_keys[1]]
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    n_remaining = n - n_train - n_val
    if n_remaining < 1:
        if n_val > 1:
            n_val -= 1
            n_remaining += 1
        else:
            n_train = max(1, n_train - 1)
            n_remaining += 1
    n_test = n_remaining
    train_keys = pair_keys[:n_train]
    val_keys = pair_keys[n_train:n_train + n_val]
    test_keys = pair_keys[n_train + n_val:]
    return train_keys, val_keys, test_keys


def create_pair_holdout_splits(df: pd.DataFrame, seed: int):
    pairs = df[["c1", "c2"]].drop_duplicates().reset_index(drop=True)
    pairs = pairs.assign(pair_key=pairs["c1"] + "::" + pairs["c2"])
    pair_keys = pairs["pair_key"].tolist()
    train_keys, val_keys, test_keys = _safe_split_pair_keys(pair_keys, seed)
    mapping = {"train": set(train_keys), "val": set(val_keys), "test": set(test_keys)}
    df = df.assign(pair_key=df["c1"] + "::" + df["c2"])
    split_frames = {}
    for split, keys in mapping.items():
        split_frames[split] = df[df["pair_key"].isin(keys)].drop(columns=["pair_key"]).reset_index(drop=True)
    return split_frames["train"], split_frames["val"], split_frames["test"]


def save_split_dfs(split_name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path):
    base = out_dir / "splits" / split_name
    base.mkdir(parents=True, exist_ok=True)
    train.to_parquet(base / "train.parquet", index=False)
    val.to_parquet(base / "val.parquet", index=False)
    test.to_parquet(base / "test.parquet", index=False)


def leakage_audit(train: pd.DataFrame, test: pd.DataFrame, split_label: str):
    chem_train = set(train["c1"]).union(train["c2"])
    chem_test = set(test["c1"]).union(test["c2"])
    overlap = chem_test.intersection(chem_train)
    pct_chem = len(overlap) / max(len(chem_test), 1) * 100
    tgt_train = set(train["target"])
    tgt_test = set(test["target"])
    pct_tgt = len(tgt_test.intersection(tgt_train)) / max(len(tgt_test), 1) * 100
    pair_train = set(zip(train["c1"], train["c2"]))
    pair_test = set(zip(test["c1"], test["c2"]))
    pct_pair = len(pair_test.intersection(pair_train)) / max(len(pair_test), 1) * 100
    return {
        "split": split_label,
        "chemicals_pct_seen": pct_chem,
        "targets_pct_seen": pct_tgt,
        "pairs_pct_seen": pct_pair,
    }
