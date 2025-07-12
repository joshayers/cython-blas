"""Benchmarks of OpenBLAS."""

import multiprocessing

from cython_blas import blis

from .gemm_base import GemmBase


class Sgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "f4"
        super().setup(size)
        blis.set_num_threads(multiprocessing.cpu_count())

    def time_sgemm(self, size: int):
        blis.sgemm(1.0, self.mat_a, self.mat_b, 0.0, self.mat_c)


class Cgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "c8"
        super().setup(size)
        blis.set_num_threads(multiprocessing.cpu_count())

    def time_cgemm(self, size: int):
        blis.cgemm(1.0, False, self.mat_a, False, self.mat_b, 0.0, self.mat_c)


class Dgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "f8"
        super().setup(size)
        blis.set_num_threads(multiprocessing.cpu_count())

    def time_dgemm(self, size: int):
        blis.dgemm(1.0, self.mat_a, self.mat_b, 0.0, self.mat_c)


class Zgemm(GemmBase):
    def setup(self, size: int):
        self.dtype = "c16"
        super().setup(size)
        blis.set_num_threads(multiprocessing.cpu_count())

    def time_zgemm(self, size: int):
        blis.zgemm(1.0, False, self.mat_a, False, self.mat_b, 0.0, self.mat_c)
