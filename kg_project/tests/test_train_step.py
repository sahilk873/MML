import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from kg_project.model.dataset import FeatureDataset
from kg_project.model.mlp import FeatureClassifier
from kg_project.model.train import train_and_evaluate


def test_train_step_runs_one_epoch():
    df = pd.DataFrame(
        {
            "c1": ["A", "A", "B", "B"],
            "c2": ["B", "C", "C", "A"],
            "target": ["T1", "T2", "T1", "T2"],
            "label": [1, 0, 1, 0],
        }
    )
    entities = sorted({*df["c1"], *df["c2"], *df["target"]})
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(entities), 8)).astype(np.float32)
    dataset = FeatureDataset(df, entity2id, embeddings)
    loader = DataLoader(dataset, batch_size=2)
    model = FeatureClassifier(7 * 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    results = train_and_evaluate(
        model,
        loader,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        max_epochs=2,
        patience=1,
    )
    assert "test_metrics" in results
    assert results["test_metrics"].get("roc_auc") is not None
