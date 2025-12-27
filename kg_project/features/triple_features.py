import numpy as np
import torch


def build_feature_vector_numpy(e1: np.ndarray, e2: np.ndarray, et: np.ndarray) -> np.ndarray:
    pair_sum = e1 + e2
    pair_diff = np.abs(e1 - e2)
    pair_prod = e1 * e2
    c1t = e1 * et
    c2t = e2 * et
    c_pair_sum = c1t + c2t
    c_pair_diff = np.abs(c1t - c2t)
    pt = pair_sum * et
    return np.concatenate([pair_sum, pair_diff, pair_prod, et, c_pair_sum, c_pair_diff, pt], axis=0)


def build_feature_vector_torch(e1: torch.Tensor, e2: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
    pair_sum = e1 + e2
    pair_diff = torch.abs(e1 - e2)
    pair_prod = e1 * e2
    c1t = e1 * et
    c2t = e2 * et
    c_pair_sum = c1t + c2t
    c_pair_diff = torch.abs(c1t - c2t)
    pt = pair_sum * et
    return torch.cat([pair_sum, pair_diff, pair_prod, et, c_pair_sum, c_pair_diff, pt], dim=-1)
