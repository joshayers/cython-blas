"""Tests of the matmul module."""

import numpy as np

from cython_blas import matmul


def test_gemm_case1():
    """Test the gemm_memview function."""
    rng = np.random.default_rng(seed=1)
    mat_a = rng.random(size=(4, 3))
    mat_b = rng.random(size=(3, 5))
    mat_c = np.zeros((4, 5), "f8")
    matmul.gemm_memview(matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans, 1.0, mat_a, mat_b, 0.0, mat_c)
    np.testing.assert_allclose(mat_c, mat_a @ mat_b)


def test_gemm_case2():
    """Test the gemm_memview function."""
    rng = np.random.default_rng(seed=1)
    mat_a = rng.random(size=(4, 3))
    mat_b = rng.random(size=(3, 5))
    mat_c = rng.random(size=(4, 5))
    mat_c_orig = mat_c.copy()
    matmul.gemm_memview(matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans, 1.0, mat_a, mat_b, 1.0, mat_c)
    np.testing.assert_allclose(mat_c, mat_a @ mat_b + mat_c_orig)
