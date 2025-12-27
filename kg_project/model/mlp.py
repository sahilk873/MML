import torch
import torch.nn as nn

from kg_project.features.triple_features import build_feature_vector_torch


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(512, 256), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        return self.out(self.mlp(x)).squeeze(-1)


class FeatureClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = MLPHead(input_dim)

    def forward(self, features: torch.Tensor):
        return self.head(features)


class IDEmbeddingModel(nn.Module):
    def __init__(self, num_entities: int, embed_dim: int):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.head = MLPHead(7 * embed_dim)

    def forward(self, c1, c2, target):
        e1 = self.entity_embeddings(c1)
        e2 = self.entity_embeddings(c2)
        et = self.entity_embeddings(target)
        features = build_feature_vector_torch(e1, e2, et)
        return self.head(features)
