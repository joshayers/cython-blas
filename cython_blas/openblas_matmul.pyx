
cimport cython


cdef extern from "cblas.h" nogil:
    cdef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor

    cdef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans
        CblasConjNoTrans

    ctypedef int blasint

    cdef void scipy_cblas_dgemm64_(
        CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
        blasint M, blasint N, blasint K,
		double alpha, const double *A, blasint lda, const double *B, blasint ldb,
        double beta, double *C, blasint ldc
    )

    cdef void scipy_cblas_zgemm64_(
         CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
         blasint M, blasint N, blasint K,
		 const void *alpha, const void *A, blasint lda, const void *B, blasint ldb,
         const void *beta, void *C, blasint ldc);


cpdef enum BLAS_Order:
    RowMajor = CblasRowMajor
    ColMajor = CblasColMajor

cpdef enum BLAS_Trans:
    NoTrans = CblasNoTrans
    Trans = CblasTrans
    ConjNoTrans = CblasConjNoTrans
    ConjTrans = CblasConjTrans


def dgemm_raw(BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, double alpha,
              const double[:, :] A, const double[:, :] B, double beta,
              double[:, :] C, blasint m, blasint n, blasint k, blasint lda, blasint ldb, blasint ldc):
    scipy_cblas_dgemm64_(
        <CBLAS_ORDER> order, <CBLAS_TRANSPOSE> ta, <CBLAS_TRANSPOSE> tb,
        m, n, k,
        alpha, &A[0, 0], lda, &B[0, 0], ldb, beta, &C[0, 0], ldc
    )


@cython.cdivision(True)
@cython.wraparound(False)
def dgemm(double alpha, const double [:, :] A, const double [:, :] B, double beta, double [:, :] C):
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

    scipy_cblas_dgemm64_(
        <CBLAS_ORDER> order_c, <CBLAS_TRANSPOSE> trans_a, <CBLAS_TRANSPOSE> trans_b,
        m, n, k,
        alpha, &A[0, 0], lda, &B[0, 0], ldb, beta, &C[0, 0], ldc
    )


def zgemm_raw(BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, double complex alpha,
              const double complex[:, :] A, const double complex[:, :] B, double complex beta,
              double complex[:, :] C, blasint m, blasint n, blasint k, blasint lda, blasint ldb, blasint ldc):
    scipy_cblas_zgemm64_(
        <CBLAS_ORDER> order, <CBLAS_TRANSPOSE> ta, <CBLAS_TRANSPOSE> tb,
        m, n, k,
        &alpha, &A[0, 0], lda, &B[0, 0], ldb, &beta, &C[0, 0], ldc
    )


