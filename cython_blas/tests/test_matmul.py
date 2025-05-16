"""Tests of the matmul module."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from cython_blas import openblas_matmul as matmul


def create_array(rng: np.random.Generator, size: tuple[int, int], dtype: str, order: str) -> npt.NDArray:
    """Create an array with the specified size, dtype, and memory order."""
    array = np.empty(size, dtype=dtype, order=order)
    if np.isrealobj(array):
        array[:] = rng.uniform(size=size)
    else:
        array.real = rng.uniform(size=size)
        array.imag = rng.uniform(size=size)
    return array


@pytest.mark.parametrize(
    ("alpha", "beta", "m", "n", "k", "a_order", "b_order", "c_order"),
    [
        (alpha, beta, 8, 9, 10, a_order, b_order, c_order)
        for alpha, beta, a_order, b_order, c_order in itertools.product(
            [0.0, 1.0, 2.1], [0.0, 1.0, 2.1], ["C", "F"], ["C", "F"], ["C", "F"]
        )
    ],
)
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
    """Test the gemm_memview function."""
    rng = np.random.default_rng(seed=1)
    mat_a = create_array(rng, (m, k), "f8", a_order)
    mat_b = create_array(rng, (k, n), "f8", b_order)
    mat_c = create_array(rng, (m, n), "f8", c_order)
    mat_c_orig = mat_c.copy()
    matmul.dgemm(alpha, mat_a, mat_b, beta, mat_c)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)
