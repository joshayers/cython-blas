"""Try it out."""

import numpy as np

from cython_blas import matmul


def random(rng: np.random.Generator, size: tuple, dtype: str, order: str) -> np.ndarray:
    """Generate random numbers, complex datatype."""
    out = np.empty(size, dtype=dtype, order=order)
    out.real = rng.random(size=size)
    out.imag = rng.random(size=size)
    return out


def main() -> None:  # noqa: PLR0915
    """Main."""
    rng = np.random.default_rng(seed=1)

    # C(cw) = alpha * A(cw) * B(cw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="F")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="F")
    mat_c = np.zeros((4, 5), "c16", order="F")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[1] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(rw) * B(cw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="C")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="F")
    mat_c = np.zeros((4, 5), "c16", order="F")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[1] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(cw) * B(rw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="F")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="C")
    mat_c = np.zeros((4, 5), "c16", order="F")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[1] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(cw) = alpha * A(rw) * B(rw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="C")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="C")
    mat_c = np.zeros((4, 5), "c16", order="F")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[1] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == m  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(ta, tb, alpha, mat_a, mat_b, beta, mat_c, m, n, k, lda, ldb, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(cw) * B(cw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="F")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="F")
    mat_c = np.zeros((4, 5), "c16", order="C")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[0] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(tb, ta, alpha, mat_b, mat_a, beta, mat_c, n, m, k, ldb, lda, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(rw) * B(cw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="C")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="F")
    mat_c = np.zeros((4, 5), "c16", order="C")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.Trans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[1] // itemsize, mat_c.strides[0] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == k  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(tb, ta, alpha, mat_b, mat_a, beta, mat_c, n, m, k, ldb, lda, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(cw) * B(rw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="F")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="C")
    mat_c = np.zeros((4, 5), "c16", order="C")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.Trans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[1] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[0] // itemsize
    assert lda == m  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(tb, ta, alpha, mat_b, mat_a, beta, mat_c, n, m, k, ldb, lda, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)

    # C(rw) = alpha * A(rw) * B(rw)
    mat_a = random(rng, size=(4, 3), dtype="c16", order="C")
    mat_b = random(rng, size=(3, 5), dtype="c16", order="C")
    mat_c = np.zeros((4, 5), "c16", order="C")
    alpha, beta = 1.0 + 1.0j, 0.0 + 0.0j
    ta, tb = matmul.BLAS_Trans.NoTrans, matmul.BLAS_Trans.NoTrans
    m, n, k = mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]
    itemsize = mat_a.dtype.itemsize
    lda, ldb, ldc = mat_a.strides[0] // itemsize, mat_b.strides[0] // itemsize, mat_c.strides[0] // itemsize
    assert lda == k  # noqa: S101
    assert ldb == n  # noqa: S101
    assert ldc == n  # noqa: S101
    mat_c_orig = mat_c.copy()
    matmul.zgemm_raw(tb, ta, alpha, mat_b, mat_a, beta, mat_c, n, m, k, ldb, lda, ldc)
    np.testing.assert_allclose(mat_c, alpha * mat_a @ mat_b + beta * mat_c_orig)


if __name__ == "__main__":
    main()
