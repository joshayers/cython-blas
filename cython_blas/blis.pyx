
from cython_blas cimport _blis


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