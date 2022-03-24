include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/profiler/include
    ${PROJECT_SOURCE_DIR}/test/include
    ${PROJECT_SOURCE_DIR}/external/include/half
)

add_test_executable(test_gemm_reduce_fp16 gemm_reduce_fp16.cpp)
target_link_libraries(test_gemm_reduce_fp16 PRIVATE host_tensor)
target_link_libraries(test_gemm_reduce_fp16 PRIVATE device_gemm_reduce_instance)
