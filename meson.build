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
