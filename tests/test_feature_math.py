import numpy as np

from magic_dmap.build_dfc_hdf5 import neurostorm_clip_starts
from magic_dmap.features import _sample_vector


def test_incomplete_trailing_window_is_excluded():
    assert neurostorm_clip_starts(95, 20, 1, 20) == [0, 20, 40, 60]
    assert neurostorm_clip_starts(19, 20, 1, 20) == []


def test_feature_cap_retains_small_vectors_and_samples_large_vectors():
    small = np.arange(4, dtype=np.float32)
    large = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(_sample_vector(small, 4), small)
    np.testing.assert_array_equal(_sample_vector(large, 4), np.array([0, 3, 6, 9], dtype=np.float32))
