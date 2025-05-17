cimport cython

from cython_blas cimport _cblas


@cython.embedsignature(True)
cpdef int openblas_get_num_threads() noexcept nogil:
    """Return the number of threads currently used by OpenBLAS."""
    return _cblas.scipy_openblas_get_num_threads64_()


@cython.embedsignature(True)
cpdef void openblas_set_num_threads(int num_threads) noexcept nogil:
    """Set the number of threads used by OpenBLAS."""
    _cblas.scipy_openblas_set_num_threads64_(num_threads)


@cython.embedsignature(True)
cpdef str openblas_get_config():
    """Return configuration information specified when OpenBLAS was compiled."""
    cdef char* cstring = _cblas.scipy_openblas_get_config64_()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


@cython.embedsignature(True)
cpdef str openblas_get_corename():
    """Return the core name currently used by OpenBLAS."""
    cdef char* cstring = _cblas.scipy_openblas_get_corename64_()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


@cython.embedsignature(True)
cpdef int openblas_get_parallel() noexcept nogil:
    """Return the type of parallelism specified when OpenBLAS was compiled.

    A return value of 0 means no parallelism.
    A return value of 1 means the normal threading model.
    A return value of 2 means the OpenMP threading model.
    """
    return _cblas.scipy_openblas_get_parallel64_()
