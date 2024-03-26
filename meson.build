project('mamba', 'objcpp',
  version : '0.1',
  default_options : [
    'warning_level=3',
    'cpp_std=c++17'])

# Disable warnings from libraries
add_project_arguments(
  '--system-header-prefix=Metal/',
  '-Wno-c99-extensions',
  language : 'objcpp')

gguf_tools_sp = subproject('gguf-tools')
gguflib_dep = gguf_tools_sp.get_variable('gguflib_dep')

deps = [
  dependency('Foundation'),
  dependency('Metal'),
  dependency('QuartzCore'),
  dependency('MetalPerformanceShaders'),
  gguflib_dep]

executable('mamba',
  sources : ['mamba.mm', 'ml.mm', 'mtl_impl.mm'],
  dependencies : deps,
  install : true)

executable('test_mamba',
  sources : ['ml.mm', 'mtl_impl.mm', 'tests/test_mamba.mm'],
  dependencies : deps,
  install : true)

xcrun = find_program('xcrun')

kernels_ir = custom_target('kernels.ir',
  input : 'kernels.metal',
  output : 'kernels.ir',
  command : [xcrun, '-sdk', 'macosx', 'metal', '-o', '@OUTPUT@', '-c', '@INPUT@'])

custom_target('kernels.metallib',
  input : kernels_ir,
  output : 'kernels.metallib',
  command : [xcrun, '-sdk', 'macosx', 'metallib', '-o', '@OUTPUT@', '@INPUT@'],
  install : true,
  install_dir : '.')
