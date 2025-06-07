"""Tests of the optimize module."""

import numpy as np
import pytest

from cython_blas import optimize


@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 15), (15, 12), (12, 16)),
        ((10, 15), (15, 12), (12, 5), (5, 16)),
    ],
)
def test_optimize_compare_to_einsum(shapes: tuple):
    """Test the optimize function."""
    mats = [optimize.Matrix(shape, "f8") for shape in shapes]
    best_path = optimize.optimize(mats)
    expression = ",".join([f"{chr(97 + i)}{chr(97 + i + 1)}" for i in range(len(shapes))])
    (_, *es_path), _ = np.einsum_path(expression, *(np.empty(shape) for shape in shapes), optimize="optimal")
    expected = tuple([i for i, _ in es_path])
    assert best_path[2] == expected
