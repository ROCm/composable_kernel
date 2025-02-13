add_executable(tile_example_gemm_basic EXCLUDE_FROM_ALL gemm_basic.cpp)
add_executable(tile_example_gemm_universal EXCLUDE_FROM_ALL universal_gemm.cpp)
target_compile_options(tile_example_gemm_universal PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)
