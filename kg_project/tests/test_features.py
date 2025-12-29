import numpy as np

from kg_project.features.triple_features import build_feature_vector_numpy


def test_feature_vectors_invariant():
    rng = np.random.default_rng(0)
    e1 = rng.standard_normal(4)
    e2 = rng.standard_normal(4)
    et = rng.standard_normal(4)
    original = build_feature_vector_numpy(e1, e2, et)
    swapped = build_feature_vector_numpy(e2, e1, et)
    d = e1.shape[0]
    # Order-invariant portions should match, while interaction slots swap.
    assert np.allclose(original[0:4 * d], swapped[0:4 * d])
    assert np.allclose(original[6 * d : 7 * d], swapped[6 * d : 7 * d])
    assert np.allclose(original[4 * d : 5 * d], swapped[5 * d : 6 * d])
    assert np.allclose(original[5 * d : 6 * d], swapped[4 * d : 5 * d])


def test_feature_vector_dimension():
    rng = np.random.default_rng(1)
    e1 = rng.standard_normal(8)
    e2 = rng.standard_normal(8)
    et = rng.standard_normal(8)
    vec = build_feature_vector_numpy(e1, e2, et)
    assert vec.shape[0] == 7 * 8


def test_feature_vector_component_ordering():
    rng = np.random.default_rng(2)
    e1 = rng.standard_normal(3)
    e2 = rng.standard_normal(3)
    et = rng.standard_normal(3)
    vec = build_feature_vector_numpy(e1, e2, et)
    d = e1.shape[0]
    pair_sum = e1 + e2
    pair_diff = np.abs(e1 - e2)
    pair_prod = e1 * e2
    c1t = e1 * et
    c2t = e2 * et
    pt = pair_sum * et

    assert np.allclose(vec[0:d], pair_sum)
    assert np.allclose(vec[d : 2 * d], pair_diff)
    assert np.allclose(vec[2 * d : 3 * d], pair_prod)
    assert np.allclose(vec[3 * d : 4 * d], et)
    assert np.allclose(vec[4 * d : 5 * d], c1t)
    assert np.allclose(vec[5 * d : 6 * d], c2t)
    assert np.allclose(vec[6 * d : 7 * d], pt)
