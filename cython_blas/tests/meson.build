subdir_path = 'cython_blas/tests'

pyx_files = []

pxd_files = []

foreach pyx_file : pyx_files
    py.extension_module(
        pyx_file,
        pyx_file + '.pyx',
        dependencies:[py_dep, np_dep],
        override_options:['cython_language=cpp'],
        install:true,
        subdir:subdir_path,
    )
endforeach

py.install_sources(
    pxd_files,
    pure:false,
    subdir:subdir_path,
)

python_sources =[
    '__init__.py',
    'test_matmul.py',
]

py.install_sources(
    python_sources,
    pure:false,
    subdir:subdir_path,
)
