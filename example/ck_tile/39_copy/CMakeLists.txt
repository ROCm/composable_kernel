add_executable(tile_example_copy EXCLUDE_FROM_ALL copy_basic.cpp)

# Impact: This flag ensures that the compiler doesn't make 
# assumptions about memory aliasing that could interfere with Composable Kernel's explicit memory access patterns.
target_compile_options(tile_example_copy PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)
