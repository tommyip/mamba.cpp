project('mamba', 'objcpp',
  version : '0.1',
  default_options : [
    'warning_level=3',
    'cpp_std=c++17'])

deps = [
  dependency('Foundation'),
  dependency('Metal'),
  dependency('QuartzCore'),
  dependency('MetalPerformanceShaders')]

src = [
  'mamba.mm',
  'ml.mm',
  'mtl_impl.mm']

executable('mamba',
  sources : src,
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
