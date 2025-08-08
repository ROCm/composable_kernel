add_custom_target(test_gemm_quantization_targets)

add_gtest_executable(test_gemm_quantization test_gemm_quantization.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_quantization PRIVATE utility device_quantization_instance)
    add_dependencies(test_gemm_quantization_targets test_gemm_quantization)
endif()

add_dependencies(test_quantization test_gemm_quantization_targets)
