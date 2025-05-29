project(
  'cython_blas',
  'cython',
  'cpp',
  meson_version: '>= 1.7.0',
  default_options: ['cpp_std=c++17'],
)

py = import('python').find_installation()

py_dep = py.dependency()
blis_dep = declare_dependency(
  dependencies: [dependency('blis', method: 'pkg-config', required: true)],
  compile_args: '-DBLIS_DISABLE_BLAS_DEFS',
)
np_dep = declare_dependency(
  dependencies: [dependency('numpy', method: 'config-tool', required: true)],
  compile_args: '-DNPY_NO_DEPRECATED_API=NPY_1_25_API_VERSION',
)
openblas_dep = dependency('scipy-openblas', method: 'pkg-config', required: true)
openmp_dep = dependency('openmp', required: true)

if get_option('coverage')
  add_project_arguments('-X', 'linetrace=True', language: 'cython')
  add_project_arguments('-DCYTHON_TRACE=1', language: 'cpp')
  add_project_arguments('-DCYTHON_TRACE_NOGIL=1', language: 'cpp')
endif

subdir('cython_blas')