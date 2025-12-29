from typing import Iterable, Set

import pandas as pd


def _gather_neighbors(kg_df: pd.DataFrame, frontier: Set[str]) -> Set[str]:
    if not frontier:
        return set()
    mask = kg_df["subject"].isin(frontier) | kg_df["object"].isin(frontier)
    if not mask.any():
        return set()
    edges = kg_df.loc[mask, ["subject", "object"]]
    # Flatten to get all adjacent nodes touching the current frontier.
    neighbors = pd.unique(edges.values.ravel(order="K"))
    return set(neighbors)


def extract_subgraph(kg_df: pd.DataFrame, seeds: Iterable[str], hops: int) -> pd.DataFrame:
    if hops <= 0:
        return kg_df
    visited: Set[str] = set(seeds)
    if not visited:
        return kg_df.iloc[0:0].copy()
    frontier: Set[str] = set(visited)
    for _ in range(hops):
        neighbors = _gather_neighbors(kg_df, frontier)
        neighbors -= visited
        if not neighbors:
            break
        visited.update(neighbors)
        frontier = neighbors
    mask = kg_df["subject"].isin(visited) & kg_df["object"].isin(visited)
    return kg_df.loc[mask].reset_index(drop=True)
