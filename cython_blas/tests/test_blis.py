"""Tests of the blis module."""

import itertools

import numpy as np
import pytest

from cython_blas import blis
from cython_blas.tests.utils import create_array

_shape_error_params_gemm = (
    ("mat_a_shape", "mat_b_shape", "mat_c_shape", "match"),
    [
        ((3, 9), (4, 4), (3, 4), r"matrix dim.*not compat.*\(3, 9\).*\(4, 4\).*\(3, 4\)"),
        ((3, 4), (4, 4), (9, 4), r"matrix dim.*not compat.*\(3, 4\).*\(4, 4\).*\(9, 4\)"),
        ((3, 4), (4, 4), (3, 9), r"matrix dim.*not compat.*\(3, 4\).*\(4, 4\).*\(3, 9\)"),
    ],
)

_real_params_gemm = (
    ("alpha", "beta", "m", "n", "k", "a_order", "b_order", "c_order"),
    [
        (alpha, beta, 8, 9, 10, a_order, b_order, c_order)
        for alpha, beta, a_order, b_order, c_order in itertools.product(
            [0.0, 1.0, 2.2], [0.0, 1.0, 2.2], ["C", "F"], ["C", "F"], ["C", "F"]
        )
    ],
)


@pytest.mark.parametrize(*_shape_error_params_gemm)
def test_dgemm_shape_error(
    mat_a_shape: tuple[int, int], mat_b_shape: tuple[int, int], mat_c_shape: tuple[int, int], match: str
):
    """Test the dgemm function, with incompatible matrix shapes."""
    alpha, beta = 1.0, 0.0
    mat_a = np.zeros(mat_a_shape, dtype="f8", order="C")
    mat_b = np.zeros(mat_b_shape, dtype="f8", order="C")
    mat_c = np.zeros(mat_c_shape, dtype="f8", order="C")
    with pytest.raises(ValueError, match=match):
        blis.dgemm(alpha, mat_a, mat_b, beta, mat_c)


@pytest.mark.parametrize(*_real_params_gemm)
def test_dgemm(  # noqa: PLR0913
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    a_order: str,
    b_order: str,
    c_order: str,
):
    """Test the dgemm function."""
    rng = np.random.default_rng(seed=1)
    mat_a = create_array(rng, (m, k), "f8", a_order)
    mat_b = create_array(rng, (k, n), "f8", b_order)
    mat_c = create_array(rng, (m, n), "f8", c_order)
    mat_c_orig = mat_c.copy()
    blis.dgemm(alpha, mat_a, mat_b, beta, mat_c)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig, atol=1e-8, rtol=1e-8)
