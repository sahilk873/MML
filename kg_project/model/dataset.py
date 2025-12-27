from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from kg_project.features.triple_features import build_feature_vector_numpy


class FeatureDataset(Dataset):
    def __init__(self, df, entity2id: Dict[str, int], embeddings: np.ndarray):
        labels = df["label"].to_numpy(dtype=np.float32)
        feature_vectors = np.stack(
            [
                build_feature_vector_numpy(
                    embeddings[entity2id[row.c1]],
                    embeddings[entity2id[row.c2]],
                    embeddings[entity2id[row.target]],
                )
                for row in df.itertuples(index=False)
            ],
            dtype=np.float32,
        )
        self.features = torch.as_tensor(feature_vectors, dtype=torch.float32).contiguous()
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }


class IDTripleDataset(Dataset):
    def __init__(self, df, entity2id: Dict[str, int]):
        self.c1 = torch.as_tensor(df["c1"].map(entity2id).to_numpy(dtype=np.int64), dtype=torch.long)
        self.c2 = torch.as_tensor(df["c2"].map(entity2id).to_numpy(dtype=np.int64), dtype=torch.long)
        self.target = torch.as_tensor(df["target"].map(entity2id).to_numpy(dtype=np.int64), dtype=torch.long)
        self.labels = torch.as_tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "c1": self.c1[idx],
            "c2": self.c2[idx],
            "target": self.target[idx],
            "label": self.labels[idx],
        }
