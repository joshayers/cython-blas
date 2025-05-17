
cimport cython

from cython_blas cimport _cblas
from cython_blas._cblas cimport blasint


cpdef enum BLAS_Order:
    RowMajor = _cblas.CblasRowMajor
    ColMajor = _cblas.CblasColMajor

cpdef enum BLAS_Trans:
    NoTrans = _cblas.CblasNoTrans
    Trans = _cblas.CblasTrans
    ConjNoTrans = _cblas.CblasConjNoTrans
    ConjTrans = _cblas.CblasConjTrans


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
def sgemm(
    float alpha,
    const float [:, :] A,
    const float [:, :] B,
    float beta,
    float [:, :] C
):
    cdef BLAS_Order order_a, order_b, order_c
    order_a = ColMajor if A.strides[0] == sizeof(float) else RowMajor
    order_b = ColMajor if B.strides[0] == sizeof(float) else RowMajor
    order_c = ColMajor if C.strides[0] == sizeof(float) else RowMajor
    cdef blasint m = A.shape[0], n = B.shape[1], k = A.shape[1]
    cdef blasint lda = A.strides[1] / sizeof(float) if order_a == ColMajor else A.strides[0] / sizeof(float)
    cdef blasint ldb = B.strides[1] / sizeof(float) if order_b == ColMajor else B.strides[0] / sizeof(float)
    cdef blasint ldc = C.strides[1] / sizeof(float) if order_c == ColMajor else C.strides[0] / sizeof(float)
    cdef BLAS_Trans trans_a = NoTrans if order_a == order_c else Trans
    cdef BLAS_Trans trans_b = NoTrans if order_b == order_c else Trans

    _cblas.scipy_cblas_sgemm64_(
        <_cblas.CBLAS_ORDER> order_c, <_cblas.CBLAS_TRANSPOSE> trans_a, <_cblas.CBLAS_TRANSPOSE> trans_b,
        m, n, k,
        alpha, &A[0, 0], lda, &B[0, 0], ldb, beta, &C[0, 0], ldc
    )


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
def dgemm(
    double alpha,
    const double [:, :] A,
    const double [:, :] B,
    double beta,
    double [:, :] C
):
    cdef BLAS_Order order_a, order_b, order_c
    order_a = ColMajor if A.strides[0] == sizeof(double) else RowMajor
    order_b = ColMajor if B.strides[0] == sizeof(double) else RowMajor
    order_c = ColMajor if C.strides[0] == sizeof(double) else RowMajor
    cdef blasint m = A.shape[0], n = B.shape[1], k = A.shape[1]
    cdef blasint lda = A.strides[1] / sizeof(double) if order_a == ColMajor else A.strides[0] / sizeof(double)
    cdef blasint ldb = B.strides[1] / sizeof(double) if order_b == ColMajor else B.strides[0] / sizeof(double)
    cdef blasint ldc = C.strides[1] / sizeof(double) if order_c == ColMajor else C.strides[0] / sizeof(double)
    cdef BLAS_Trans trans_a = NoTrans if order_a == order_c else Trans
    cdef BLAS_Trans trans_b = NoTrans if order_b == order_c else Trans

    _cblas.scipy_cblas_dgemm64_(
        <_cblas.CBLAS_ORDER> order_c, <_cblas.CBLAS_TRANSPOSE> trans_a, <_cblas.CBLAS_TRANSPOSE> trans_b,
        m, n, k,
        alpha, &A[0, 0], lda, &B[0, 0], ldb, beta, &C[0, 0], ldc
    )

@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
def cgemm(
    float complex alpha,
    bint conjugate_a,
    const float complex [:, :] A,
    bint conjugate_b,
    const float complex [:, :] B,
    float complex beta,
    float complex [:, :] C,
):
    cdef BLAS_Order order_a, order_b, order_c
    order_a = ColMajor if A.strides[0] == sizeof(float complex) else RowMajor
    order_b = ColMajor if B.strides[0] == sizeof(float complex) else RowMajor
    order_c = ColMajor if C.strides[0] == sizeof(float complex) else RowMajor
    cdef blasint m = A.shape[0], n = B.shape[1], k = A.shape[1]
    cdef blasint lda = (A.strides[1] / sizeof(float complex)
                        if order_a == ColMajor else A.strides[0] / sizeof(float complex))
    cdef blasint ldb = (B.strides[1] / sizeof(float complex)
                        if order_b == ColMajor else B.strides[0] / sizeof(float complex))
    cdef blasint ldc = (C.strides[1] / sizeof(float complex)
                        if order_c == ColMajor else C.strides[0] / sizeof(float complex))
    cdef BLAS_Trans trans_a
    if conjugate_a:
        trans_a = ConjNoTrans if order_a == order_c else ConjTrans
    else:
        trans_a = NoTrans if order_a == order_c else Trans
    cdef BLAS_Trans trans_b
    if conjugate_b:
        trans_b = ConjNoTrans if order_b == order_c else ConjTrans
    else:
        trans_b = NoTrans if order_b == order_c else Trans

    _cblas.scipy_cblas_cgemm64_(
        <_cblas.CBLAS_ORDER> order_c, <_cblas.CBLAS_TRANSPOSE> trans_a, <_cblas.CBLAS_TRANSPOSE> trans_b,
        m, n, k,
        &alpha, &A[0, 0], lda, &B[0, 0], ldb, &beta, &C[0, 0], ldc
    )


@cython.cdivision(True)
@cython.embedsignature(True)
@cython.wraparound(False)
def zgemm(
    double complex alpha,
    bint conjugate_a,
    const double complex [:, :] A,
    bint conjugate_b,
    const double complex [:, :] B,
    double complex beta,
    double complex [:, :] C,
):
    cdef BLAS_Order order_a, order_b, order_c
    order_a = ColMajor if A.strides[0] == sizeof(double complex) else RowMajor
    order_b = ColMajor if B.strides[0] == sizeof(double complex) else RowMajor
    order_c = ColMajor if C.strides[0] == sizeof(double complex) else RowMajor
    cdef blasint m = A.shape[0], n = B.shape[1], k = A.shape[1]
    cdef blasint lda = (A.strides[1] / sizeof(double complex)
                        if order_a == ColMajor else A.strides[0] / sizeof(double complex))
    cdef blasint ldb = (B.strides[1] / sizeof(double complex)
                        if order_b == ColMajor else B.strides[0] / sizeof(double complex))
    cdef blasint ldc = (C.strides[1] / sizeof(double complex)
                        if order_c == ColMajor else C.strides[0] / sizeof(double complex))
    cdef BLAS_Trans trans_a
    if conjugate_a:
        trans_a = ConjNoTrans if order_a == order_c else ConjTrans
    else:
        trans_a = NoTrans if order_a == order_c else Trans
    cdef BLAS_Trans trans_b
    if conjugate_b:
        trans_b = ConjNoTrans if order_b == order_c else ConjTrans
    else:
        trans_b = NoTrans if order_b == order_c else Trans

    _cblas.scipy_cblas_zgemm64_(
        <_cblas.CBLAS_ORDER> order_c, <_cblas.CBLAS_TRANSPOSE> trans_a, <_cblas.CBLAS_TRANSPOSE> trans_b,
        m, n, k,
        &alpha, &A[0, 0], lda, &B[0, 0], ldb, &beta, &C[0, 0], ldc
    )
