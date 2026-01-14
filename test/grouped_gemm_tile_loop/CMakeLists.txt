# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_grouped_gemm_tile_loop)

if (CK_USE_XDL OR CK_USE_WMMA)
    add_gtest_executable(test_grouped_gemm_tile_loop_vanilla test_grouped_gemm_tile_loop.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_grouped_gemm_tile_loop_vanilla PRIVATE utility device_grouped_gemm_tile_loop_instance)
        add_dependencies(test_grouped_gemm_tile_loop test_grouped_gemm_tile_loop_vanilla)
    endif()

    add_gtest_executable(test_grouped_gemm_tile_loop_multiply test_grouped_gemm_tile_loop_multiply.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_grouped_gemm_tile_loop_multiply PRIVATE utility device_grouped_gemm_tile_loop_instance)
        add_dependencies(test_grouped_gemm_tile_loop test_grouped_gemm_tile_loop_multiply)
    endif()
endif()