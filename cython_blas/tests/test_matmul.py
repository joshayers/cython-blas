"""Tests of the matmul module."""

import numpy as np

from cython_blas import matmul


def test_gemm():
    """Test the gemm_memview function."""
    rng = np.random.default_rng(seed=1)
    mat_a = rng.random(size=(4, 3))
    mat_b = rng.random(size=(3, 5))
    mat_c = np.zeros((4, 5), "f8")
    matmul.gemm_memview(110, 110, 1.0, mat_a, mat_b, 0.0, mat_c)
    np.testing.assert_allclose(mat_c, mat_a @ mat_b)
