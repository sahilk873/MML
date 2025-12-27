from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from pykeen.datasets import EagerDataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def train_pykeen(
    triples_df,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    model_name: str,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    out_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, Dict[str, object]]:
    training_triples = triples_df[["subject", "predicate", "object"]].values
    tf = TriplesFactory.from_labeled_triples(
        training_triples,
        entity_to_id=entity2id,
        relation_to_id=relation2id,
    )
    dataset = EagerDataset(training=tf, testing=tf)
    pipeline_result = pipeline(
        dataset=dataset,
        model=model_name.capitalize(),
        model_kwargs={"embedding_dim": embed_dim},
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size},
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        random_seed=seed,
        device=device,
    )
    embeddings = pipeline_result.model.entity_representations[0]().detach().cpu().numpy()
    np.save(out_dir / "node_embeddings.npy", embeddings)
    stats = {"entities": embeddings.shape[0], "dim": embeddings.shape[1], "epochs": epochs}
    return embeddings, stats
