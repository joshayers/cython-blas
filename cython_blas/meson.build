subdir_path = 'cython_blas'

cython_args = []

openblas_pyx_files = [
    'openblas',
]
foreach pyx_file : openblas_pyx_files
    py.extension_module(
        pyx_file,
        pyx_file + '.pyx',
        dependencies: [
            py_dep,
            np_dep,
            openblas_dep,
        ],
        override_options: ['cython_language=cpp'],
        cython_args: cython_args,
        subdir: subdir_path,
        limited_api: py_limited_api_version,
        install: true,
    )
endforeach

blis_pyx_files = [
    'blis',
]
foreach pyx_file : blis_pyx_files
    py.extension_module(
        pyx_file,
        pyx_file + '.pyx',
        dependencies: [
            py_dep,
            np_dep,
            blis_dep,
            openmp_dep,
        ],
        override_options: ['cython_language=cpp'],
        cython_args: cython_args,
        cpp_args: ['-Wno-unused-parameter'], # suppress warnings from blis.h
        subdir: subdir_path,
        limited_api: py_limited_api_version,
        install: true,
    )
endforeach