from pathlib import Path

import pandas as pd

from kg_project.io.csv_loader import read_label_csvs
from kg_project.io.parquet_loader import read_kg
from kg_project.preprocess.normalize_pairs import normalize_label_pairs


def test_parquet_loader_smoke(tmp_path: Path):
    node_df = pd.DataFrame(
        {
            "id": ["C1", "C2"],
            "category": ["chem", "chem_b"],
            "labels": [["L1"], ["L2"]],
            "all_categories": [["A"], ["B"]],
        }
    )
    part = tmp_path / "part-00000.snappy.parquet"
    node_df.to_parquet(part)
    kg_df = read_kg([part])
    assert not kg_df.empty
    assert set(["subject", "predicate", "object"]).issubset(kg_df.columns)
    assert kg_df["subject"].dtype == object
    assert kg_df["predicate"].dtype == object


def test_label_normalization(tmp_path: Path):
    pos = pd.DataFrame({"source_primary": ["A"], "source_secondary": ["B"], "target": ["T"]})
    neg = pd.DataFrame({"source_primary": ["B"], "source_secondary": ["A"], "target": ["T"]})
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"
    pos["extra"] = ["x"]
    neg["extra"] = ["y"]
    pos.to_csv(pos_path, index=False)
    neg.to_csv(neg_path, index=False)
    combined = read_label_csvs(tmp_path, pos_path.name, neg_path.name)
    normalized, stats = normalize_label_pairs(combined.drop(columns=["extra"]))
    assert normalized.shape[0] == 1
    assert stats["conflicts_resolved"] == 1
    assert stats["positives"] == 1
