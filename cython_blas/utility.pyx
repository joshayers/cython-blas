cimport cython

from cython_blas cimport _cblas


@cython.embedsignature(True)
cpdef int openblas_get_num_threads() noexcept nogil:
    return _cblas.scipy_openblas_get_num_threads64_()


@cython.embedsignature(True)
cpdef void openblas_set_num_threads(int num_threads) noexcept nogil:
    _cblas.scipy_openblas_set_num_threads64_(num_threads)


cpdef str openblas_get_config():
    cdef char* cstring = _cblas.scipy_openblas_get_config64_()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


cpdef str openblas_get_corename():
    """Return the core name currently used by OpenBLAS."""
    cdef char* cstring = _cblas.scipy_openblas_get_corename64_()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


@cython.embedsignature(True)
cpdef int openblas_get_parallel() noexcept nogil:
    return _cblas.scipy_openblas_get_parallel64_()
