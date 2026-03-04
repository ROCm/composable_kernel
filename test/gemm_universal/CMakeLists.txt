# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


# NOTE: We test for XDL/WMMA support here instead of relying on the usual pattern matching in the parent CMakeLists. This is necessary
# as these tests are universal and dont have "xdl" or "wmma" in their name to signify their target arch. But they will fail to link
# the instance library if there's no instances present for the current arch.
if (CK_USE_XDL OR CK_USE_WMMA) 
    add_gtest_executable(test_gemm_universal_fp16 test_gemm_universal_fp16.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_universal_fp16 PRIVATE utility device_gemm_universal_instance)
    endif()

    add_gtest_executable(test_gemm_universal_fp8 test_gemm_universal_fp8.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_universal_fp8 PRIVATE utility device_gemm_universal_instance)
    endif()

    add_gtest_executable(test_gemm_universal_bf16 test_gemm_universal_bf16.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_universal_bf16 PRIVATE utility device_gemm_universal_instance)
    endif()
endif()
