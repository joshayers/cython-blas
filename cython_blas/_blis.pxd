
cdef extern from "blis.h" nogil:

    ctypedef int arch_t

    const char* bli_info_get_int_type_size_str()
    const char* bli_info_get_version_str()
    arch_t bli_arch_query_id()
    const char* bli_arch_string(arch_t id)