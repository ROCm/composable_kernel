set(GEMM_BIAS_ADD_SOURCES
    gemm_bias_add_xdl_fp16.cpp
    gemm_bias_add_fp16.cpp
)
add_executable(example_gemm_bias_add_xdl_fp16   ${GEMM_BIAS_ADD_SOURCES})
target_link_libraries(example_gemm_bias_add_xdl_fp16 PRIVATE utility)
