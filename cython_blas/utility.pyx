from cython_blas cimport _cblas


@cython.embedsignature(True)
cpdef int openblas_get_num_threads() noexcept nogil:
    return _cblas.scipy_openblas_get_num_threads64_()


@cython.embedsignature(True)
cpdef void openblas_set_num_threads(int num_threads) noexcept nogil:
    _cblas.scipy_openblas_set_num_threads64_(num_threads)


@cython.embedsignature(True)
cpdef int openblas_get_parallel() noexcept nogil:
    return _cblas.scipy_openblas_get_parallel64_()
