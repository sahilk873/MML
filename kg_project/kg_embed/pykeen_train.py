from pathlib import Path
from typing import Dict, Tuple

import logging
import numpy as np
import torch
from pykeen.datasets import EagerDataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


logger = logging.getLogger(__name__)


def ensure_real_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Convert complex/float64 embeddings to contiguous float32 arrays."""
    if np.iscomplexobj(embeddings):
        embeddings = np.concatenate([embeddings.real, embeddings.imag], axis=1)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    return np.ascontiguousarray(embeddings)


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

    def _make_eval_factory(factory: TriplesFactory, rng_seed: int) -> TriplesFactory:
        """Create a tiny held-out factory so PyKEEN evaluation has triples to score."""
        total_triples = factory.num_triples
        if total_triples == 0:
            raise ValueError("PyKEEN received an empty triple set")
        eval_count = min(total_triples, max(1, int(0.05 * total_triples)))
        if eval_count == total_triples:
            return factory
        generator = torch.Generator()
        generator.manual_seed(rng_seed)
        indices = torch.randperm(total_triples, generator=generator)[:eval_count]
        eval_triples = factory.mapped_triples.index_select(0, indices).clone()
        return factory.clone_and_exchange_triples(eval_triples, keep_metadata=False)

    eval_factory = _make_eval_factory(tf, seed)
    dataset = EagerDataset(training=tf, testing=eval_factory, validation=eval_factory)

    def _make_training_kwargs(target_device: torch.device, requested_batch: int):
        kwargs = {"num_epochs": epochs, "batch_size": requested_batch}
        if target_device.type == "cpu":
            # On CPU runs fall back to conservative manual batching to avoid the costly
            # automatic slicing search that frequently fails for complex models.
            effective_batch = min(requested_batch, 1024)
            sub_batch = min(effective_batch, 512)
            if effective_batch < requested_batch:
                logger.info(
                    "Reducing PyKEEN batch size from %d to %d for CPU training to avoid slicing",
                    requested_batch,
                    effective_batch,
                )
            kwargs = {
                "num_epochs": epochs,
                "batch_size": effective_batch,
                "sub_batch_size": max(1, sub_batch),
            }
        return kwargs

    def _run_pipeline(run_device: torch.device, kwargs):
        return pipeline(
            dataset=dataset,
            model=model_name.capitalize(),
            model_kwargs={"embedding_dim": embed_dim},
            training_kwargs=kwargs,
            optimizer="Adam",
            optimizer_kwargs={"lr": lr},
            random_seed=seed,
            device=run_device,
            use_testing_data=False,
        )

    def _execute_with_retry(run_device: torch.device, kwargs):
        try:
            return _run_pipeline(run_device, kwargs)
        except MemoryError:
            fallback_batch = max(1, kwargs["batch_size"] // 2)
            if fallback_batch == kwargs["batch_size"]:
                raise
            logger.warning(
                "PyKEEN MemoryError with batch size %d on %s; retrying with %d",
                kwargs["batch_size"],
                run_device,
                fallback_batch,
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["batch_size"] = fallback_batch
            retry_kwargs["sub_batch_size"] = max(1, fallback_batch // 2)
            return _run_pipeline(run_device, retry_kwargs)

    training_kwargs = _make_training_kwargs(device, batch_size)

    try:
        pipeline_result = _execute_with_retry(device, training_kwargs)
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda":
            raise
        logger.warning(
            "CUDA out of memory while training PyKEEN on %s; retrying on CPU with conservative settings",
            device,
        )
        torch.cuda.empty_cache()
        cpu_device = torch.device("cpu")
        cpu_kwargs = _make_training_kwargs(cpu_device, batch_size)
        pipeline_result = _execute_with_retry(cpu_device, cpu_kwargs)
    embeddings = pipeline_result.model.entity_representations[0]().detach().cpu().numpy()
    embeddings = ensure_real_embeddings(embeddings)
    np.save(out_dir / "node_embeddings.npy", embeddings)
    stats = {"entities": embeddings.shape[0], "dim": embeddings.shape[1], "epochs": epochs}
    return embeddings, stats
