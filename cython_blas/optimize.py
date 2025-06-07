"""Optimize matrix multiplication."""

import itertools
from typing import Self

import numpy as np


class Matrix:
    """Matrix class."""

    def __init__(self, shape: tuple, dtype: str) -> None:
        """Initialize the class."""
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other: Self) -> bool:
        """Check for equality."""
        if self.shape != other.shape:
            return False
        return self.dtype == other.dtype

    def calc_flops(self, other: Self) -> tuple[int, Self]:
        """Calculate the flops due to multiplying two matrices."""
        if isinstance(other, MultiMatrix):
            flops = 0
            mats = []
            for mat2 in other.mats:
                flops_i, mats_i = self.calc_flops(mat2)
                flops += flops_i
                mats.append(mats_i)
            return flops, MultiMatrix(mats)
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

    def calc_flops(self, other: Self | Matrix) -> tuple[int, Self]:
        """Calculate the flops."""
        flops = 0
        mats = []
        if isinstance(other, MultiMatrix):
            for mat1, mat2 in zip(self.mats, other.mats, strict=True):
                flops_i, mats_i = mat1.calc_flops(mat2)
                flops += flops_i
                mats.append(mats_i)
        else:
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


if __name__ == "__main__":
    mat1 = Matrix((10, 15), "f8")
    mat2 = Matrix((15, 12), "f8")
    mat3 = Matrix((12, 16), "f8")
    best_path, worst_path = optimize([mat1, mat2, mat3])
    print()
    print(best_path)
    print(worst_path)
    path, descr = np.einsum_path(
        "ab,bc,cd", np.empty((10, 15)), np.empty((15, 12)), np.empty((12, 16)), optimize="optimal"
    )
    print(path)
    print(descr)

    mat1 = Matrix((10, 15), "f8")
    mat2 = Matrix((15, 12), "f8")
    mat3 = Matrix((12, 5), "f8")
    mat4 = Matrix((5, 16), "f8")
    best_path, worst_path = optimize([mat1, mat2, mat3, mat4])
    print()
    print(best_path)
    print(worst_path)
    path, descr = np.einsum_path(
        "ab,bc,cd,de", np.empty((10, 15)), np.empty((15, 12)), np.empty((12, 5)), np.empty((5, 16)), optimize="optimal"
    )
    print(path)
    print(descr)

    mat1 = Matrix((10, 15), "f8")
    mat2 = Matrix((15, 12), "f8")
    mat3 = Matrix((12, 5), "f8")
    mat4 = Matrix((5, 16), "f8")
    mat5 = Matrix((16, 15), "f8")
    best_path, worst_path = optimize([mat1, mat2, mat3, mat4, mat5])
    print()
    print(best_path)
    print(worst_path)
    path, descr = np.einsum_path(
        "ab,bc,cd,de,ef",
        np.empty((10, 15)),
        np.empty((15, 12)),
        np.empty((12, 5)),
        np.empty((5, 16)),
        np.empty((16, 15)),
        optimize="optimal",
    )
    print(path)
    print(descr)

    mat1 = Matrix((2, 2), "f8")
    mat2 = Matrix((2, 5), "f8")
    mat3 = Matrix((5, 2), "f8")
    best_path, worst_path = optimize([mat1, mat2, mat3])
    print()
    print(best_path)
    print(worst_path)
