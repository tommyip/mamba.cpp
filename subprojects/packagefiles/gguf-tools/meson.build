project('gguf-tools', 'c',
  default_options : ['c_std=c11'])

gguflib_sources = [
  'gguflib.c',
  'fp16.c']
inc = include_directories('.')
gguflib = library('gguflib', gguflib_sources,
  include_directories : inc,
  install : true)

gguflib_dep = declare_dependency(
  link_with : gguflib)
