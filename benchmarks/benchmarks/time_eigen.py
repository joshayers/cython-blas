"""Benchmarks of Eigen."""

import multiprocessing

from cython_blas import eigen

from .gemm_base import GemmBase


class Sgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "f4"
        super().setup(size)
        eigen.set_num_threads(multiprocessing.cpu_count())

    def time_sgemm(self, size: int):
        eigen.sgemm(1.0, self.mat_a, self.mat_b, 0.0, self.mat_c)


class Dgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "f8"
        super().setup(size)
        eigen.set_num_threads(multiprocessing.cpu_count())

    def time_dgemm(self, size: int):
        eigen.dgemm(1.0, self.mat_a, self.mat_b, 0.0, self.mat_c)
