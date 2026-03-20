from numpy cimport int64_t


cdef extern from "eigen.hpp" nogil:

    enum StorageOptions:
        RowMajor
        ColMajor

    ctypedef int row_maj_t "RowMajor"
    ctypedef int col_maj_t "ColMajor"

    void eigen_gemm[T, order_a, order_b, order_c](
        const T alpha,
        const T* A_p,
        const T* B_p,
        const T beta,
        T* C_p,
        const int64_t m,
        const int64_t n,
        const int64_t k
    )

    void setNbThreads(int v)

    int nbThreads()

    void map_tests(double* A_p, int, int, int, int)
