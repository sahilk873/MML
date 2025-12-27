import pandas as pd

from kg_project.splits.splitters import create_pair_holdout_splits


def test_pair_holdout_separates_pairs():
    df = pd.DataFrame(
        {
            "c1": ["A", "A", "B", "B"],
            "c2": ["B", "B", "C", "C"],
            "target": ["T1", "T2", "T1", "T2"],
            "label": [1, 0, 1, 0],
        }
    )
    train, val, test = create_pair_holdout_splits(df, seed=0)
    train_pairs = set(zip(train["c1"], train["c2"]))
    test_pairs = set(zip(test["c1"], test["c2"]))
    assert train_pairs.isdisjoint(test_pairs)
