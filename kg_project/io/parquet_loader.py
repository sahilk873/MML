from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def expand_list_relation(node_df: pd.DataFrame, column: str, predicate: str) -> pd.DataFrame:
    if column not in node_df:
        return pd.DataFrame(columns=["subject", "predicate", "object"])
    subset = node_df[["id", column]].explode(column)
    subset = subset.dropna(subset=[column])
    subset = subset[subset[column].astype(bool)]
    subset = subset.rename(columns={"id": "subject", column: "object"})
    subset["predicate"] = predicate
    return subset[["subject", "predicate", "object"]]


def expand_str_relation(node_df: pd.DataFrame, column: str, predicate: str) -> pd.DataFrame:
    if column not in node_df:
        return pd.DataFrame(columns=["subject", "predicate", "object"])
    subset = node_df[["id", column]].dropna(subset=[column])
    subset = subset[subset[column].astype(bool)]
    subset = subset.rename(columns={"id": "subject", column: "object"})
    subset["predicate"] = predicate
    return subset[["subject", "predicate", "object"]]


def node_frame_to_triples(node_df: pd.DataFrame) -> pd.DataFrame:
    relations = [
        expand_str_relation(node_df, "category", "biolink:has_category"),
        expand_list_relation(node_df, "equivalent_identifiers", "biolink:same_as"),
        expand_list_relation(node_df, "all_categories", "biolink:has_category"),
        expand_list_relation(node_df, "labels", "biolink:has_label"),
        expand_list_relation(node_df, "upstream_data_source", "biolink:has_upstream_data_source"),
        expand_str_relation(
            node_df, "international_resource_identifier", "biolink:has_international_resource_identifier"
        ),
    ]
    return pd.concat(relations, ignore_index=True) if relations else pd.DataFrame(columns=["subject", "predicate", "object"])


def list_kg_parts(src_dir: Path, glob_pattern: str) -> List[Path]:
    files = sorted(src_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No KG files found for pattern {glob_pattern}")
    return files


def read_kg(parts: Iterable[Path]) -> pd.DataFrame:
    triples = []
    for part in parts:
        node_df = pd.read_parquet(part)
        triples.append(node_frame_to_triples(node_df))
    kg_df = pd.concat(triples, ignore_index=True)
    kg_df = kg_df.dropna(subset=["subject", "predicate", "object"])
    kg_df = kg_df.astype(str)
    kg_df = kg_df.drop_duplicates()
    return kg_df


def log_kg_stats(kg_df: pd.DataFrame):
    edges = len(kg_df)
    subjects = kg_df["subject"].nunique()
    predicates = kg_df["predicate"].nunique()
    objects = kg_df["object"].nunique()
    entities = pd.unique(kg_df[["subject", "object"]].values.ravel()).size
    df = {
        "edges": edges,
        "unique_subjects": subjects,
        "unique_predicates": predicates,
        "unique_objects": objects,
        "unique_entities": entities,
    }
    return df


def persist_kg(kg_df: pd.DataFrame, out_dir: Path):
    path = out_dir / "kg_edges.parquet"
    kg_df.to_parquet(path, index=False)
