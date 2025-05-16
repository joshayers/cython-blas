"""Try it out."""

import numpy as np

from cython_blas import openblas_matmul as matmul


def main() -> None:  # noqa: PLR0915
    """Main."""
    rng = np.random.default_rng(seed=1)

    # C(cw) = alpha * A(cw) * B(cw)
    mat_a = np.asfortranarray(rng.random(size=(4, 3)))
    mat_b = np.asfortranarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="F")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.ColMajor, matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[1] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(rw) * B(cw)
    mat_a = np.ascontiguousarray(rng.random(size=(4, 3)))
    mat_b = np.asfortranarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="F")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.ColMajor, matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[1] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(cw) * B(rw)
    mat_a = np.asfortranarray(rng.random(size=(4, 3)))
    mat_b = np.ascontiguousarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="F")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.ColMajor, matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[1] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(rw) * B(rw)
    mat_a = np.ascontiguousarray(rng.random(size=(4, 3)))
    mat_b = np.ascontiguousarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="F")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.ColMajor, matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[1] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(cw) * B(cw)
    mat_a = np.asfortranarray(rng.random(size=(4, 3)))
    mat_b = np.asfortranarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="C")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.RowMajor, matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[0] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(rw) * B(cw)
    mat_a = np.ascontiguousarray(rng.random(size=(4, 3)))
    mat_b = np.asfortranarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="C")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.RowMajor, matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[0] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(cw) * B(rw)
    mat_a = np.asfortranarray(rng.random(size=(4, 3)))
    mat_b = np.ascontiguousarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="C")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.RowMajor, matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[0] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(rw) * B(rw)
    mat_a = np.ascontiguousarray(rng.random(size=(4, 3)))
    mat_b = np.ascontiguousarray(rng.random(size=(3, 5)))
    mat_c = np.zeros((4, 5), "f8", order="C")
    alpha, beta = 1.0, 0.0
    order, ta, tb = matmul.BLAS_Order.RowMajor, matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[0] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.dgemm_raw(order, ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)


if __name__ == "__main__":
    main()
