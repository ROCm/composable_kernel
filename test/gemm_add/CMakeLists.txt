# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Implements test instances for MultipleD with xdl and wmma support.

# NOTE: We test for XDL/WMMA support here instead of relying on the usual pattern matching in the parent CMakeLists. This is necessary
# as these tests are universal and dont have "xdl" or "wmma" in their name to signify their target arch. But they will fail to link
# the instance library if there's no instances present for the current arch.
if (CK_USE_XDL OR CK_USE_WMMA) 
    add_gtest_executable(test_gemm_add test_gemm_add.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_add PRIVATE utility device_gemm_add_instance)
    endif()

    add_gtest_executable(test_gemm_add_relu test_gemm_add_relu.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_add_relu PRIVATE utility device_gemm_add_instance device_gemm_add_relu_instance)
    endif()

    add_gtest_executable(test_gemm_add_silu test_gemm_add_silu.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_add_silu PRIVATE utility device_gemm_add_instance device_gemm_add_silu_instance)
    endif()

    add_gtest_executable(test_gemm_add_fastgelu test_gemm_add_fastgelu.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_add_fastgelu PRIVATE utility device_gemm_add_fastgelu_instance)
    endif()
endif()

add_gtest_executable(test_gemm_fastgelu_wmma test_gemm_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_fastgelu_wmma PRIVATE utility device_gemm_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_add_add_fastgelu_wmma test_gemm_add_add_fastgelu_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_add_fastgelu_wmma PRIVATE utility device_gemm_add_add_fastgelu_instance)
endif()

add_gtest_executable(test_gemm_multiply_multiply_wmma test_gemm_multiply_multiply_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multiply_multiply_wmma PRIVATE utility device_gemm_multiply_multiply_instance)
endif()

add_gtest_executable(test_gemm_add_multiply_wmma test_gemm_add_multiply_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_add_multiply_wmma PRIVATE utility device_gemm_add_multiply_instance)
endif()

add_gtest_executable(test_gemm_multiply_add_wmma test_gemm_multiply_add_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multiply_add_wmma PRIVATE utility device_gemm_multiply_add_instance)
endif()

add_gtest_executable(test_gemm_bilinear_wmma test_gemm_bilinear_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_bilinear_wmma PRIVATE utility device_gemm_bilinear_instance)
endif()