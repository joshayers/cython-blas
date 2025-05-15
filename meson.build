project(
  'cython_blas',
  'cython',
  'cpp',
  meson_version: '>= 1.7.0',
  default_options: [
    'warning_level=2',
  ],
)

py = import('python').find_installation()

py_dep = py.dependency()
np_dep = declare_dependency(
  dependencies: [dependency('numpy', method: 'config-tool', required: true)],
  compile_args: '-DNPY_NO_DEPRECATED_API=NPY_1_25_API_VERSION',
)

if get_option('coverage')
  add_project_arguments('-X', 'linetrace=True', language: 'cython')
  add_project_arguments('-DCYTHON_TRACE=1', language: 'cpp')
  add_project_arguments('-DCYTHON_TRACE_NOGIL=1', language: 'cpp')
endif

subdir('cython_blas')