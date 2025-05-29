
cimport cython

from cython_blas cimport _blis
from cython_blas._blis cimport bli_gemm


def get_int_type_size():
    cdef const char* cstring = _blis.bli_info_get_int_type_size_str()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


def get_version():
    cdef const char* cstring = _blis.bli_info_get_version_str()
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


def get_arch():
    cdef _blis.arch_t id = _blis.bli_arch_query_id()
    cdef const char* cstring = _blis.bli_arch_string(id)
    cdef bytes bstring = cstring
    return bstring.decode('ascii')


@cython.cdivision(True)
def dgemm(
    double alpha,
    const double [:, :] A,
    const double [:, :] B,
    double beta,
    double [:, :] C
):
    cdef _blis.obj_t bli_a, bli_b, bli_c, bli_alpha, bli_beta

    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &alpha,
        &bli_alpha,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        A.shape[0],
        A.shape[1],
        <void*> &A[0, 0],
        A.strides[0] // sizeof(double),
        A.strides[1] // sizeof(double),
        &bli_a,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        B.shape[0],
        B.shape[1],
        <void*> &B[0, 0],
        B.strides[0] // sizeof(double),
        B.strides[1] // sizeof(double),
        &bli_b,
    )
    _blis.bli_obj_create_1x1_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        &beta,
        &bli_beta,
    )
    _blis.bli_obj_create_with_attached_buffer(
        _blis.BLIS_DOUBLE,
        C.shape[0],
        C.shape[1],
        <void*> &C[0, 0],
        C.strides[0] // sizeof(double),
        C.strides[1] // sizeof(double),
        &bli_c,
    )

    bli_gemm(&bli_alpha, &bli_a, &bli_b, &bli_beta, &bli_c)