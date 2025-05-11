"""Matrix multiplication."""

from scipy.linalg.cython_blas cimport dgemm, sgemm


cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'

ctypedef fused floating:
    float
    double


cdef void _gemm(BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, int m, int n,
                int k, floating alpha, const floating *A, int lda, const floating *B,
                int ldb, floating beta, floating *C, int ldc) noexcept nogil:
    """C := alpha * op(A).op(B) + beta * C"""
    # TODO: Remove the pointer casts below once SciPy uses const-qualification.
    # See: https://github.com/scipy/scipy/issues/14262
    cdef:
        char ta_ = ta
        char tb_ = tb
    if order == BLAS_Order.RowMajor:
        if floating is float:
            sgemm(&tb_, &ta_, &n, &m, &k, &alpha, <float*>B,
                  &ldb, <float*>A, &lda, &beta, C, &ldc)
        else:
            dgemm(&tb_, &ta_, &n, &m, &k, &alpha, <double*>B,
                  &ldb, <double*>A, &lda, &beta, C, &ldc)
    else:
        if floating is float:
            sgemm(&ta_, &tb_, &m, &n, &k, &alpha, <float*>A,
                  &lda, <float*>B, &ldb, &beta, C, &ldc)
        else:
            dgemm(&ta_, &tb_, &m, &n, &k, &alpha, <double*>A,
                  &lda, <double*>B, &ldb, &beta, C, &ldc)


cpdef gemm_memview(BLAS_Trans ta, BLAS_Trans tb, floating alpha,
                   const floating[:, :] A, const floating[:, :] B, floating beta,
                   floating[:, :] C):
    cdef:
        int m = A.shape[0] if ta == BLAS_Trans.NoTrans else A.shape[1]
        int n = B.shape[1] if tb == BLAS_Trans.NoTrans else B.shape[0]
        int k = A.shape[1] if ta == BLAS_Trans.NoTrans else A.shape[0]
        int lda, ldb, ldc
        BLAS_Order order = (
            BLAS_Order.ColMajor if A.strides[0] == A.itemsize else BLAS_Order.RowMajor
        )

    if order == BLAS_Order.RowMajor:
        lda = k if ta == BLAS_Trans.NoTrans else m
        ldb = n if tb == BLAS_Trans.NoTrans else k
        ldc = n
    else:
        lda = m if ta == BLAS_Trans.NoTrans else k
        ldb = k if tb == BLAS_Trans.NoTrans else n
        ldc = m

    _gemm(order, ta, tb, m, n, k, alpha, &A[0, 0],
          lda, &B[0, 0], ldb, beta, &C[0, 0], ldc)