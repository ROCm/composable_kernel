add_executable(tile_tutorial_naive_gemm EXCLUDE_FROM_ALL practice_gemm.cpp)

target_compile_options(tile_tutorial_naive_gemm PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)

add_dependencies(tutorials tile_tutorial_naive_gemm)