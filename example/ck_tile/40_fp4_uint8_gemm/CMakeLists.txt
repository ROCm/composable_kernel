add_executable(tile_example_fp4_uint8_gemm EXCLUDE_FROM_ALL fp4_uint8_gemm.cpp)

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)
target_compile_options(tile_example_fp4_uint8_gemm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
