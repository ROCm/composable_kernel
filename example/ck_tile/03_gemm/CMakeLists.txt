add_executable(tile_example_gemm_basic gemm_basic.cpp)
add_executable(tile_example_gemm_universal universal_gemm.cpp)

target_compile_options(tile_example_gemm_universal PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)
