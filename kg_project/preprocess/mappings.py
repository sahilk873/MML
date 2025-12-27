import json
from pathlib import Path

import pandas as pd


def build_id_mappings(kg_df: pd.DataFrame, out_dir: Path):
    entities = pd.unique(kg_df[["subject", "object"]].values.ravel())
    entities = sorted(entities)
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relations = sorted(kg_df["predicate"].unique())
    relation2id = {relation: idx for idx, relation in enumerate(relations)}
    (out_dir / "entity2id.json").write_text(json.dumps(entity2id, indent=2))
    (out_dir / "relation2id.json").write_text(json.dumps(relation2id, indent=2))
    return entity2id, relation2id
