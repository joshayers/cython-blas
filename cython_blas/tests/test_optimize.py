"""Tests of the optimize module."""

import re

import numpy as np
import pytest

from cython_blas import optimize

np_einsum_regex = re.compile(r"Optimized FLOP count:[ ]*([\d\.e\-\+]+)\s", re.IGNORECASE)


def parse_optimized_flop_count(string: str) -> int:
    """Parse the optimized FLOP count from np.einsum_path."""
    match = np_einsum_regex.search(string)
    return int(float(match.group(1)))


@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 15), (15, 12), (12, 16)),
        ((10, 15), (15, 12), (12, 5), (5, 16)),
        ((10, 15), (15, 22), (22, 5), (5, 16)),
    ],
)
def test_optimize_compare_to_einsum(shapes: tuple):
    """Test the optimize function."""
    mats = [optimize.Matrix(shape, "f8") for shape in shapes]
    best_path = optimize.optimize(mats)
    expression = ",".join([f"{chr(97 + i)}{chr(97 + i + 1)}" for i in range(len(shapes))])
    (_, *es_path), es_descr = np.einsum_path(expression, *(np.empty(shape) for shape in shapes), optimize="optimal")
    es_flop_count = parse_optimized_flop_count(es_descr)
    assert best_path[1] == es_flop_count - 1
    print(best_path[1])
