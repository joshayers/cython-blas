
cimport cython
cimport numpy as np
from numpy cimport ndarray

from cython_blas cimport _blis
from cython_blas._blis cimport dim_t, obj_t


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
cpdef dgemm(
    double alpha,
    const double [:, :] A,
    const double [:, :] B,
    double beta,
    double [:, :] C
):
    r"""Matrix multiplication of double precision matrices.

    .. math::
        C = \alpha A B + \beta C
    """
    cdef dim_t m = A.shape[0], n = B.shape[1], k = A.shape[1]
    if B.shape[0] != k or C.shape[0] != m or C.shape[1] != n:
        msg = (
            "matrix dimensions not compatible: "
            f"({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0], B.shape[1]}) = ({C.shape[0]}, {C.shape[1]})"
        )
        raise ValueError(msg)

    cdef obj_t bli_a, bli_b, bli_c, bli_alpha, bli_beta
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &alpha,
        &bli_alpha,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        m,
        k,
        <void*> &A[0, 0],
        A.strides[0] // sizeof(double),
        A.strides[1] // sizeof(double),
        &bli_a,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        k,
        n,
        <void*> &B[0, 0],
        B.strides[0] // sizeof(double),
        B.strides[1] // sizeof(double),
        &bli_b,
    )
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &beta,
        &bli_beta,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        m,
        n,
        <void*> &C[0, 0],
        C.strides[0] // sizeof(double),
        C.strides[1] // sizeof(double),
        &bli_c,
    )

    _blis.bli_gemm(&bli_alpha, &bli_a, &bli_b, &bli_beta, &bli_c)


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
cpdef zgemm(
    double complex alpha,
    bint conjugate_a,
    const double complex [:, :] A,
    bint conjugate_b,
    const double complex [:, :] B,
    double complex beta,
    double complex [:, :] C,
):
    r"""Matrix multiplication of double precision complex matrices.

    .. math::
        C = \alpha A B + \beta C

    If `conjugate_a` is True, then matrix :math:`A` is implicitly conjugated before performing
    the multiplication. Similarly for `conjugate_b`.
    """
    cdef dim_t m = A.shape[0], n = B.shape[1], k = A.shape[1]
    if B.shape[0] != k or C.shape[0] != m or C.shape[1] != n:
        msg = (
            "matrix dimensions not compatible: "
            f"({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0], B.shape[1]}) = ({C.shape[0]}, {C.shape[1]})"
        )
        raise ValueError(msg)

    cdef obj_t bli_a, bli_b, bli_c, bli_alpha, bli_beta
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        &alpha,
        &bli_alpha,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        m,
        k,
        <void*> &A[0, 0],
        A.strides[0] // sizeof(double complex),
        A.strides[1] // sizeof(double complex),
        &bli_a,
    )
    if conjugate_a:
        _blis.bli_obj_set_conj(_blis.BLIS_CONJUGATE, &bli_a)
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        k,
        n,
        <void*> &B[0, 0],
        B.strides[0] // sizeof(double complex),
        B.strides[1] // sizeof(double complex),
        &bli_b,
    )
    if conjugate_b:
        _blis.bli_obj_set_conj(_blis.BLIS_CONJUGATE, &bli_b)
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        &beta,
        &bli_beta,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        m,
        n,
        <void*> &C[0, 0],
        C.strides[0] // sizeof(double complex),
        C.strides[1] // sizeof(double complex),
        &bli_c,
    )

    _blis.bli_gemm(&bli_alpha, &bli_a, &bli_b, &bli_beta, &bli_c)


def gemm(
    double alpha,
    ndarray A,
    ndarray B,
    double beta,
    ndarray C,
):
    if np.PyArray_NDIM(A) != 2 or np.PyArray_NDIM(B) != 2 or np.PyArray_NDIM(C) != 2:
        msg = 'matrices A, B, and C must be two-dimensional'
        raise ValueError(msg)
    cdef dim_t m = np.PyArray_DIM(A, 0)
    cdef dim_t n = np.PyArray_DIM(B, 1)
    cdef dim_t k = np.PyArray_DIM(A, 1)
    if np.PyArray_DIM(B, 0) != k or np.PyArray_DIM(C, 0) != m or np.PyArray_DIM(C, 1) != n:
        msg = (
            "matrix dimensions not compatible: "
            f"({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0], B.shape[1]}) = ({C.shape[0]}, {C.shape[1]})"
        )
        raise ValueError(msg)

    cdef obj_t bli_a, bli_b, bli_c, bli_alpha, bli_beta
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &alpha,
        &bli_alpha,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        m,
        k,
        np.PyArray_DATA(A),
        np.PyArray_STRIDE(A, 0) // np.PyArray_ITEMSIZE(A),
        np.PyArray_STRIDE(A, 1) // np.PyArray_ITEMSIZE(A),
        &bli_a,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        k,
        n,
        np.PyArray_DATA(B),
        np.PyArray_STRIDE(B, 0) // np.PyArray_ITEMSIZE(B),
        np.PyArray_STRIDE(B, 1) // np.PyArray_ITEMSIZE(B),
        &bli_b,
    )
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &beta,
        &bli_beta,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DCOMPLEX,
        m,
        n,
        np.PyArray_DATA(C),
        np.PyArray_STRIDE(C, 0) // np.PyArray_ITEMSIZE(C),
        np.PyArray_STRIDE(C, 1) // np.PyArray_ITEMSIZE(C),
        &bli_c,
    )

    _blis.bli_gemm(&bli_alpha, &bli_a, &bli_b, &bli_beta, &bli_c)


def get_int_type_size() -> str:
    """Return the integer size used by BLIS."""
    cdef const char* cstring = _blis.bli_info_get_int_type_size_str()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


def get_version() -> str:
    """Return the version of BLIS."""
    cdef const char* cstring = _blis.bli_info_get_version_str()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


def get_arch() -> str:
    """Return the architecture name currently used by BLIS."""
    cdef _blis.arch_t id = _blis.bli_arch_query_id()
    cdef const char* cstring = _blis.bli_arch_string(id)
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


cpdef void set_num_threads(dim_t n_threads) noexcept nogil:
    """Set the number of threads used by BLIS."""
    _blis.bli_thread_set_num_threads(n_threads)


cpdef dim_t get_num_threads() noexcept nogil:
    """Return the number of threads currently used by BLIS."""
    return _blis.bli_thread_get_num_threads()
