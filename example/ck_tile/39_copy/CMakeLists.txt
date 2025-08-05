add_executable(tile_example_copy EXCLUDE_FROM_ALL copy_basic.cpp)

target_compile_options(tile_example_copy PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)