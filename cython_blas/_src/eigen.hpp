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

enum ContiguousType
{
    Not_Contig,
    C_Contig,
    F_Contig
};

template <ContiguousType Contig, typename T>
static auto map2(T const *A_p, const Index shape0, const Index shape1, const Index stride0, const Index stride1)
{
    if constexpr (Contig == Not_Contig)
    {
        Stride<Dynamic, Dynamic> stride = Stride<Dynamic, Dynamic>(stride0, stride1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<const matrix_t, Unaligned, Stride<Dynamic, Dynamic>> A(A_p, shape0, shape1, stride);
        return A;
    }
    if constexpr (Contig == C_Contig)
    {
        Stride<Dynamic, 1> stride = Stride<Dynamic, 1>(stride0, 1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<const matrix_t, Unaligned, Stride<Dynamic, 1>> A(A_p, shape0, shape1, stride);
        return A;
    }
    if constexpr (Contig == F_Contig)
    {
        Stride<1, Dynamic> stride = Stride<1, Dynamic>(1, stride1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<const matrix_t, Unaligned, Stride<1, Dynamic>> A(A_p, shape0, shape1, stride);
        return A;
    }
}

template <ContiguousType Contig, typename T>
static auto map2(T *A_p, const Index shape0, const Index shape1, const Index stride0, const Index stride1)
{
    if constexpr (Contig == Not_Contig)
    {
        Stride<Dynamic, Dynamic> stride = Stride<Dynamic, Dynamic>(stride0, stride1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<matrix_t, Unaligned, Stride<Dynamic, Dynamic>> A(A_p, shape0, shape1, stride);
        return A;
    }
    if constexpr (Contig == C_Contig)
    {
        Stride<Dynamic, 1> stride = Stride<Dynamic, 1>(stride0, 1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<matrix_t, Unaligned, Stride<Dynamic, 1>> A(A_p, shape0, shape1, stride);
        return A;
    }
    if constexpr (Contig == F_Contig)
    {
        Stride<1, Dynamic> stride = Stride<1, Dynamic>(1, stride1);
        typedef Matrix<T, Dynamic, Dynamic, RowMajor> matrix_t;
        Map<matrix_t, Unaligned, Stride<1, Dynamic>> A(A_p, shape0, shape1, stride);
        return A;
    }
}

template <typename T>
static void map_tests(T const *A_p, const Index shape0, const Index shape1, const Index stride0, const Index stride1)
{
    if (stride0 > 1 && stride1 > 1)
    {
        const ContiguousType Contig = Not_Contig;
        std::cout << "Not_Contig" << "\n";
        auto A = map2<Contig, T>(A_p, shape0, shape1, stride0, stride1);
        std::cout << A << std::endl;
        std::cout << A(1, 1) << std::endl;
        // A(1, 1) = -333.0;
    }
    else if (stride0 == 1 && stride1 > 1)
    {
        const ContiguousType Contig = F_Contig;
        std::cout << "F_Contig" << "\n";
        auto A = map2<Contig, T>(A_p, shape0, shape1, stride0, stride1);
        std::cout << A << std::endl;
        std::cout << A(1, 1) << std::endl;
        // A(1, 1) = -333.0;
    }
    else if (stride0 > 1 && stride1 == 1)
    {
        const ContiguousType Contig = C_Contig;
        std::cout << "C_Contig" << "\n";
        auto A = map2<Contig, T>(A_p, shape0, shape1, stride0, stride1);
        std::cout << A << std::endl;
        std::cout << A(1, 1) << std::endl;
        // A(1, 1) = -333.0;
    }
}

/*
#include <iostream>
#include <string>

// The return type is deduced automatically based on the 'Size' parameter
template <typename T, int Size>
auto processData(T value) {
    if constexpr (Size == 64) {
        // Return type becomes std::string
        return std::string("Large payload: ") + std::to_string(value);
    } else {
        // Return type becomes the original type T (e.g., int, double)
        return value * 2;
    }
}

int main() {
    // 1. Deduces return type as 'int'
    auto result1 = processData<int, 32>(10);
    std::cout << result1 << " (Type: " << typeid(result1).name() << ")\n";

    // 2. Deduces return type as 'std::string'
    auto result2 = processData<int, 64>(10);
    std::cout << result2 << " (Type: " << typeid(result2).name() << ")\n";
}

*/
