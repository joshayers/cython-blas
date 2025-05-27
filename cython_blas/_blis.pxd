
cdef extern from "blis.h" nogil:

    const char* bli_info_get_int_type_size_str()
    const char* bli_info_get_version_str()
