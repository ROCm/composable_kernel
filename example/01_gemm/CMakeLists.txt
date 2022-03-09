set(GEMM_XDL_FP16_SOURCE gemm_xdl_fp16.cpp)
set(GEMM_XDL_BF16_SOURCE gemm_xdl_bf16.cpp)
set(GEMM_XDL_INT8_SOURCE gemm_xdl_int8.cpp)

add_executable(gemm_xdl_fp16 ${GEMM_XDL_FP16_SOURCE})
add_executable(gemm_xdl_bf16 ${GEMM_XDL_BF16_SOURCE})
add_executable(gemm_xdl_int8 ${GEMM_XDL_INT8_SOURCE})

target_link_libraries(gemm_xdl_fp16 PRIVATE host_tensor)
target_link_libraries(gemm_xdl_bf16 PRIVATE host_tensor)
target_link_libraries(gemm_xdl_int8 PRIVATE host_tensor)
