cimport cython
cimport numpy as np
from numpy cimport int64_t

import numpy as np

np.import_array()


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
cpdef int sgemm(
    float alpha,
    const float [:, :] A,
    const float [:, :] B,
    float beta,
    float [:, :] C
) except -1:
    cdef StorageOptions order_a = ColMajor if A.strides[0] == sizeof(float) else RowMajor
    cdef StorageOptions order_b = ColMajor if B.strides[0] == sizeof(float) else RowMajor
    cdef StorageOptions order_c = ColMajor if C.strides[0] == sizeof(float) else RowMajor
    cdef int64_t m = A.shape[0], n = B.shape[1], k = A.shape[1]
    if B.shape[0] != k or C.shape[0] != m or C.shape[1] != n:
        msg = (
            "matrix dimensions not compatible: "
            f"({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0], B.shape[1]}) = ({C.shape[0]}, {C.shape[1]})"
        )
        raise ValueError(msg)
    if order_a == RowMajor and A.strides[1] != sizeof(float):
        msg = "matrix A must be contiguous along one dimension"
        raise ValueError(msg)
    if order_b == RowMajor and B.strides[1] != sizeof(float):
        msg = "matrix B must be contiguous along one dimension"
        raise ValueError(msg)
    if order_c == RowMajor and C.strides[1] != sizeof(float):
        msg = "matrix C must be contiguous along one dimension"
        raise ValueError(msg)
    with nogil, cython.boundscheck(False):
        if order_a == RowMajor and order_b == RowMajor and order_c == RowMajor:
            eigen_gemm[float, row_maj_t, row_maj_t, row_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == RowMajor and order_b == RowMajor and order_c == ColMajor:
            eigen_gemm[float, row_maj_t, row_maj_t, col_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == RowMajor and order_b == ColMajor and order_c == RowMajor:
            eigen_gemm[float, row_maj_t, col_maj_t, row_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == RowMajor and order_b == ColMajor and order_c == ColMajor:
            eigen_gemm[float, row_maj_t, col_maj_t, col_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == ColMajor and order_b == RowMajor and order_c == RowMajor:
            eigen_gemm[float, col_maj_t, row_maj_t, row_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == ColMajor and order_b == RowMajor and order_c == ColMajor:
            eigen_gemm[float, col_maj_t, row_maj_t, col_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == ColMajor and order_b == ColMajor and order_c == RowMajor:
            eigen_gemm[float, col_maj_t, col_maj_t, row_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
        elif order_a == ColMajor and order_b == ColMajor and order_c == ColMajor:
            eigen_gemm[float, col_maj_t, col_maj_t, col_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
    return 0


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
cpdef int dgemm(
    double alpha,
    const double [:, :] A,
    const double [:, :] B,
    double beta,
    double [:, :] C
) except -1:
    cdef int64_t m = A.shape[0], n = B.shape[1], k = A.shape[1]
    with nogil, cython.boundscheck(False):
        eigen_gemm[double, row_maj_t, row_maj_t, row_maj_t](alpha, &A[0, 0], &B[0, 0], beta, &C[0, 0], m, n, k)
    return 0


@cython.embedsignature(True)
cpdef void set_num_threads(int n_threads) noexcept nogil:
    """Set the number of threads used by Eigen."""
    setNbThreads(n_threads)


@cython.embedsignature(True)
cpdef int get_num_threads() noexcept nogil:
    """Return the number of threads currently used by Eigen."""
    return nbThreads()


def mappp():
    print()
    A = np.arange(24).astype('f8')
    A.shape = (4, 6)
    A = np.ascontiguousarray(A)
    print(A.strides)
    print()

    print(A)
    print(A[1, 1])
    print()
    map_tests[double](<double*>np.PyArray_DATA(A), A.shape[0], A.shape[1], A.strides[0] // 8, A.strides[1] // 8)
    print()
    print(A[1, 1])
    print('\n\n')

    A = np.arange(24).astype('f8')
    A.shape = (4, 6)
    A = np.ascontiguousarray(A)
    B = A[::2, ::2]
    print(B)
    print(B[1, 1])
    print()
    map_tests[double](<double*>np.PyArray_DATA(B), B.shape[0], B.shape[1], B.strides[0] // 8, B.strides[1] // 8)
    print()
    print(B[1, 1])
    print('\n\n')

    A = np.arange(24).astype('f8')
    A.shape = (4, 6)
    A = np.ascontiguousarray(A)
    C = A[::2, ::3]
    print(C)
    print(C[1, 1])
    print()
    map_tests[double](<double*>np.PyArray_DATA(C), C.shape[0], C.shape[1], C.strides[0] // 8, C.strides[1] // 8)
    print()
    print(C[1, 1])
    print('\n\n')

    A = np.arange(24).astype('f8')
    A.shape = (4, 6)
    A = np.ascontiguousarray(A)
    D = A.T
    print(D)
    print(D[1, 1])
    print()
    map_tests[double](<double*>np.PyArray_DATA(D), D.shape[0], D.shape[1], D.strides[0] // 8, D.strides[1] // 8)
    print()
    print(D[1, 1])
    print('\n\n')
