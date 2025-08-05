add_executable(tile_example_practice_gemm EXCLUDE_FROM_ALL practice_gemm.cpp)

target_compile_options(tile_example_practice_gemm PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)