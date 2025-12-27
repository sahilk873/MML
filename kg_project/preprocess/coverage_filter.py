from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def filter_labels_by_entities(
    df: pd.DataFrame, entity_set: Iterable[str], out_dir: Path
) -> Dict[str, object]:
    entity_set = set(entity_set)
    missing_mask = ~df["c1"].isin(entity_set) | ~df["c2"].isin(entity_set) | ~df["target"].isin(entity_set)
    dropped = df[missing_mask].copy()
    reasons: List[str] = []
    for _, row in dropped.iterrows():
        missing = []
        for col in ["c1", "c2", "target"]:
            if row[col] not in entity_set:
                missing.append(col)
        reasons.append(";".join(missing))
    prefixes = Counter()
    if not dropped.empty:
        dropped = dropped.assign(missing_reason=reasons)
        dropped.to_csv(out_dir / "dropped_missing_entities.csv", index=False)
        for entity in pd.concat([dropped["c1"], dropped["c2"], dropped["target"]]):
            if entity not in entity_set:
                prefix = entity.split(":", 1)[0] if ":" in entity else entity
                prefixes[prefix] += 1
    kept = df[~missing_mask].reset_index(drop=True)
    kept.to_parquet(out_dir / "labels_filtered.parquet", index=False)
    drop_pct = (1 - len(kept) / len(df)) * 100 if len(df) else 0.0
    stats = {
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_pct": drop_pct,
        "top_prefixes": prefixes.most_common(5),
    }
    return {"kept_df": kept, "dropped_df": dropped, "stats": stats}
