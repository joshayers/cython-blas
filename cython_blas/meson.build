subdir_path = 'cython_blas'

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
        install: true,
        subdir: subdir_path,
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
        cpp_args: ['-Wno-unused-parameter'], # suppress warnings from blis.h
        install: true,
        subdir: subdir_path,
    )
endforeach

pxd_files = [
    '_blis.pxd',
    '_cblas.pxd',
]

py.install_sources(
    pxd_files,
    pure: false,
    subdir: subdir_path,
)

python_sources = [
    '__init__.py',
]

py.install_sources(
    python_sources,
    pure: false,
    subdir: subdir_path,
)

subdir('tests')