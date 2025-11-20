add_gtest_executable(test_gemm_bias_add_reduce_fp16 test_gemm_bias_add_reduce_fp16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_bias_add_reduce_fp16 PRIVATE utility device_gemm_bias_add_reduce_instance)
endif()
