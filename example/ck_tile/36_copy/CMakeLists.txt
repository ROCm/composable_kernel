add_executable(test_copy_kernel EXCLUDE_FROM_ALL test_copy.cpp)
target_compile_options(test_copy_kernel PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)