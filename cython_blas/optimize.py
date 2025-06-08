"""Optimize matrix multiplication."""

import itertools
from typing import TypeVar

Matrix = TypeVar("Matrix", bound="Matrix")
MultiMatrix = TypeVar("MultiMatrix", bound="MultiMatrix")


class Matrix:
    """Matrix class."""

    def __init__(self, shape: tuple, dtype: str) -> None:
        """Initialize the class."""
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other: Matrix | MultiMatrix) -> bool:
        """Check for equality."""
        if self.shape != other.shape:
            return False
        return self.dtype == other.dtype

    def calc_flops(self, other: Matrix | MultiMatrix) -> tuple[int, Matrix | MultiMatrix]:
        """Calculate the flops due to multiplying two matrices."""
        if isinstance(other, MultiMatrix):
            return other.calc_flops_left(self)
        dtype_mult = {"f4": 1, "f8": 2, "c8": 2, "c16": 4}
        mult = max(dtype_mult[self.dtype], dtype_mult[other.dtype])
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError
        flops = mult * m * k * n
        return flops, Matrix((m, n), self.dtype)


class MultiMatrix:
    """MultiMatrix class."""

    def __init__(self, mats: list[Matrix]) -> None:
        """Initialize the class."""
        self.mats = mats

    def _calc_flops_multimatrix(self, other: MultiMatrix) -> tuple[int, MultiMatrix]:
        """Calculate the flops when multiplying two MultiMatrix instances."""
        flops = 0
        mats = []
        for mat1, mat2 in zip(self.mats, other.mats, strict=True):
            flops_i, mats_i = mat1.calc_flops(mat2)
            flops += flops_i
            mats.append(mats_i)
        return flops, MultiMatrix(mats)

    def calc_flops_left(self, other: Matrix) -> tuple[int, MultiMatrix]:
        """Calculate the flops for other @ self."""
        flops = 0
        mats = []
        for mat2 in self.mats:
            flops_i, mats_i = other.calc_flops(mat2)
            flops += flops_i
            mats.append(mats_i)
        return flops, MultiMatrix(mats)

    def calc_flops(self, other: Matrix | MultiMatrix) -> tuple[int, Matrix | MultiMatrix]:
        """Calculate the flops for self @ other."""
        if isinstance(other, MultiMatrix):
            return self._calc_flops_multimatrix(other)
        flops = 0
        mats = []
        for mat1 in self.mats:
            flops_i, mats_i = mat1.calc_flops(other)
            flops += flops_i
            mats.append(mats_i)
        return flops, MultiMatrix(mats)


def optimize(mats: list[Matrix]) -> tuple[int, int, tuple[int]]:
    """Optimize the order of matrix multiplication."""
    n_mats = len(mats)
    paths = itertools.product(*(range(i) for i in range(n_mats - 1, 0, -1)))
    best_path = (-1, 2**63, ())
    for i, path in enumerate(paths):
        flops = 0
        mats_i = list(mats)
        for pair in path:
            flops_i, remaining = mats_i[pair].calc_flops(mats_i[pair + 1])
            flops += flops_i
            mats_i = [*mats_i[:pair], remaining, *mats_i[pair + 2 :]]
        if flops < best_path[1]:
            best_path = (i, flops, path)
    return best_path
