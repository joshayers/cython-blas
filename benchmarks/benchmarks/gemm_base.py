"""Benchmarks of generalized matrix multiply (gemm) functions."""

import numpy as np


class GemmBase:
    param_names = ["size"]
    params = [[100, 1000, 5000]]

    def setup(self, size: int):
        self.mat_a = np.ones((size, size), self.dtype, order="C")
        self.mat_b = np.ones((size, size), self.dtype, order="C")
        self.mat_c = np.ones((size, size), self.dtype, order="C")
