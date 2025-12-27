import numpy as np

from kg_project.features.triple_features import build_feature_vector_numpy


def test_feature_vectors_invariant():
    rng = np.random.default_rng(0)
    e1 = rng.standard_normal(4)
    e2 = rng.standard_normal(4)
    et = rng.standard_normal(4)
    original = build_feature_vector_numpy(e1, e2, et)
    swapped = build_feature_vector_numpy(e2, e1, et)
    assert np.allclose(original, swapped)


def test_feature_vector_dimension():
    rng = np.random.default_rng(1)
    e1 = rng.standard_normal(8)
    e2 = rng.standard_normal(8)
    et = rng.standard_normal(8)
    vec = build_feature_vector_numpy(e1, e2, et)
    assert vec.shape[0] == 7 * 8
