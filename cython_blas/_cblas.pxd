
cdef extern from "cblas.h" nogil:

    void scipy_openblas_set_num_threads64_(int num_threads)
    int scipy_openblas_get_num_threads64_()

    char* scipy_openblas_get_config64_()
    char* scipy_openblas_get_corename64_()
    int scipy_openblas_get_parallel64_()

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
