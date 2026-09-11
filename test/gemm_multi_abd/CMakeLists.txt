# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NOTE: We test for XDL/WMMA support here instead of relying on the usual pattern matching in the parent CMakeLists. This is necessary
# as these tests are universal and dont have "xdl" or "wmma" in their name to signify their target arch. But they will fail to link
# the instance library if there's no instances present for the current arch.
if (CK_USE_XDL OR CK_USE_WMMA) 
    add_gtest_executable(test_gemm_multi_abd test_gemm_multi_abd.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_multi_abd PRIVATE utility device_gemm_multi_abd_instance)
    endif()
endif()