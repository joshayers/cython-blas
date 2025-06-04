"""Tests of the blis module."""

import itertools

import numpy as np
import pytest

from cython_blas import blis
from cython_blas.tests.utils import conjugate_if, create_array

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

_complex_params_gemm = (
    ("alpha", "conjugate_a", "beta", "conjugate_b", "m", "n", "k", "a_order", "b_order", "c_order"),
    [
        (alpha, conjugate_a, beta, conjugate_b, 8, 9, 10, a_order, b_order, c_order)
        for alpha, conjugate_a, beta, conjugate_b, a_order, b_order, c_order in itertools.product(
            [0.0 + 0.0j, 1.0 + 1.2j, 2.1 + 1.0j],
            [True, False],
            [0.0 + 0.0j, 1.0 + 1.2j, 2.1 + 1.0j],
            [True, False],
            ["C", "F"],
            ["C", "F"],
            ["C", "F"],
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


@pytest.mark.parametrize(*_shape_error_params_gemm)
def test_zgemm_shape_error(
    mat_a_shape: tuple[int, int], mat_b_shape: tuple[int, int], mat_c_shape: tuple[int, int], match: str
):
    """Test the zgemm function, with incompatible matrix shapes."""
    alpha, beta = 1.0, 0.0
    conjugate_a, conjugate_b = False, False
    mat_a = np.zeros(mat_a_shape, dtype="c16", order="C")
    mat_b = np.zeros(mat_b_shape, dtype="c16", order="C")
    mat_c = np.zeros(mat_c_shape, dtype="c16", order="C")
    with pytest.raises(ValueError, match=match):
        blis.zgemm(alpha, conjugate_a, mat_a, conjugate_b, mat_b, beta, mat_c)


@pytest.mark.parametrize(*_complex_params_gemm)
def test_zgemm(  # noqa: PLR0913
    alpha: complex,
    conjugate_a: bool,
    beta: complex,
    conjugate_b: bool,
    m: int,
    n: int,
    k: int,
    a_order: str,
    b_order: str,
    c_order: str,
):
    """Test the zgemm function."""
    rng = np.random.default_rng(seed=1)
    mat_a = create_array(rng, (m, k), "c16", a_order)
    mat_b = create_array(rng, (k, n), "c16", b_order)
    mat_c = create_array(rng, (m, n), "c16", c_order)
    mat_c_orig = mat_c.copy()
    blis.zgemm(alpha, conjugate_a, mat_a, conjugate_b, mat_b, beta, mat_c)
    expected = alpha * conjugate_if(mat_a, conjugate_a) @ conjugate_if(mat_b, conjugate_b) + beta * mat_c_orig
    np.testing.assert_allclose(mat_c, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    (
        "alpha",
        "conjugate_a",
        "beta",
        "conjugate_b",
        "m",
        "n",
        "k",
        "a_dtype",
        "b_dtype",
        "c_dtype",
        "a_order",
        "b_order",
        "c_order",
    ),
    [
        (alpha, conjugate_a, beta, conjugate_b, 8, 9, 10, a_dtype, b_dtype, c_dtype, a_order, b_order, c_order)
        for (
            alpha,
            conjugate_a,
            beta,
            conjugate_b,
            a_dtype,
            b_dtype,
            c_dtype,
            a_order,
            b_order,
            c_order,
        ) in itertools.product(
            [0.0, 1.0, 2.1],
            [True, False],
            [0.0 + 0.0j, 1.0 + 1.2j, 2.1 + 1.0j],
            [True, False],
            ["f4", "f8", "c8", "c16"],
            ["f4", "f8", "c8", "c16"],
            ["f4", "f8", "c8", "c16"],
            ["C", "F"],
            ["C", "F"],
            ["C", "F"],
        )
    ],
)
def test_gemm(  # noqa: PLR0913
    alpha: float,
    conjugate_a: bool,
    beta: complex,
    conjugate_b: bool,
    m: int,
    n: int,
    k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    a_order: str,
    b_order: str,
    c_order: str,
):
    """Test the gemm function."""
    rng = np.random.default_rng(seed=1)
    mat_a = create_array(rng, (m, k), a_dtype, a_order)
    mat_b = create_array(rng, (k, n), b_dtype, b_order)
    mat_c = create_array(rng, (m, n), c_dtype, c_order)
    mat_c_orig = mat_c.copy()
    blis.gemm(alpha, conjugate_a, mat_a, conjugate_b, mat_b, beta, mat_c)
    expected = alpha * conjugate_if(mat_a, conjugate_a) @ conjugate_if(mat_b, conjugate_b) + beta * mat_c_orig
    expected = expected.real if c_dtype in ("f4", "f8") else expected
    atol = 5e-6 if c_dtype in ("f4", "c8") else 1e-6
    rtol = 5e-6 if c_dtype in ("f4", "c8") else 1e-6
    np.testing.assert_allclose(mat_c, expected, atol=atol, rtol=rtol)
