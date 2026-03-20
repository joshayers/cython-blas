#pragma once

#include <Eigen/Dense>
#include "numpy/ndarraytypes.h"

using namespace Eigen;

template <typename T, int order_a, int order_b, int order_c>
static void
eigen_gemm(const T alpha, T const *A_p, T const *B_p, const T beta, T *C_p,
           const npy_int64 m, const npy_int64 n, const npy_int64 k)
{
    Map<const Matrix<T, Dynamic, Dynamic, order_a>> A(A_p, m, k);
    Map<const Matrix<T, Dynamic, Dynamic, order_b>> B(B_p, k, n);
    Map<Matrix<T, Dynamic, Dynamic, order_b>> C(C_p, m, n);
    if (alpha == 0.0)
    {
        C *= (1.0 + beta);
    }
    else if (beta == 0.0)
    {
        C.noalias() = alpha * A * B;
    }
    else
    {
        C = alpha * A * B + beta * C;
    }
}

#include <iostream>

static void
map_tests(double *A_p, int shape0, int shape1, int stride0, int stride1)
{
    auto stride = Stride<Dynamic, Dynamic>(stride0, stride1);
    typedef Matrix<double, Dynamic, Dynamic, RowMajor> matrix_t;
    Map<const matrix_t, Unaligned, Stride<Dynamic, Dynamic>> A(A_p, shape0, shape1, stride);

    std::cout << A << std::endl;
}