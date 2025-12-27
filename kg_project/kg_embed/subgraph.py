from collections import defaultdict
from typing import Iterable

import pandas as pd


def extract_subgraph(kg_df: pd.DataFrame, seeds: Iterable[str], hops: int) -> pd.DataFrame:
    if hops <= 0:
        return kg_df
    neighbors = defaultdict(set)
    for _, row in kg_df.iterrows():
        neighbors[row["subject"]].add(row["object"])
        neighbors[row["object"]].add(row["subject"])
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(neighbors.get(node, []))
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    sub_df = kg_df[kg_df["subject"].isin(visited) & kg_df["object"].isin(visited)].reset_index(drop=True)
    return sub_df
