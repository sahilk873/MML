from typing import Dict, Tuple

import pandas as pd


def normalize_label_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    combined = df[["source_primary", "source_secondary", "target", "label"]].copy()
    combined = combined.fillna("")
    combined["c1"] = combined[["source_primary", "source_secondary"]].min(axis=1)
    combined["c2"] = combined[["source_primary", "source_secondary"]].max(axis=1)
    cleaned = combined[["c1", "c2", "target", "label"]].copy()
    grouped = cleaned.groupby(["c1", "c2", "target"])
    conflicts = grouped["label"].nunique()
    conflict_count = int((conflicts > 1).sum())
    normalized = grouped["label"].max().reset_index()
    stats = {
        "examples": len(normalized),
        "positives": int(normalized["label"].sum()),
        "negatives": int(len(normalized) - normalized["label"].sum()),
        "unique_chemicals": pd.unique(normalized[["c1", "c2"]].values.ravel()).size,
        "unique_targets": normalized["target"].nunique(),
        "conflicts_resolved": conflict_count,
    }
    return normalized, stats
