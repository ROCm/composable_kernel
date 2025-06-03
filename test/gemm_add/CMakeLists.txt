add_gtest_executable(test_gemm_add_xdl test_gemm_add_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_xdl PRIVATE utility device_gemm_add_instance)
endif()

add_gtest_executable(test_gemm_add_relu_xdl test_gemm_add_relu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_relu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_relu_instance)
endif()

add_gtest_executable(test_gemm_add_silu_xdl test_gemm_add_silu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_silu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_silu_instance)
endif()

add_gtest_executable(test_gemm_add_fastgelu_xdl test_gemm_add_fastgelu_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_fastgelu_xdl PRIVATE utility device_gemm_add_instance device_gemm_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_add_fastgelu_wmma test_gemm_add_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_fastgelu_wmma PRIVATE utility device_gemm_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_add_multiply_wmma test_gemm_add_multiply_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_multiply_wmma PRIVATE utility device_gemm_add_multiply_instance)
endif()
