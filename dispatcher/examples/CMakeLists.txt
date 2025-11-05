# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Single CK Tile kernel example (follows tile_engine pattern)
# Includes one kernel via -include flag
set(KERNEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x32_4x4x1_32x32x16.hpp")

if(EXISTS "${KERNEL_HEADER}")
    add_executable(single_tile_kernel_example
        single_tile_kernel_example.cpp
    )
    
    target_link_libraries(single_tile_kernel_example PRIVATE
        ck_tile_dispatcher
    )
    
    # Add include paths
    target_include_directories(single_tile_kernel_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
    )
    
    # Use -include to force include the kernel header (tile_engine pattern)
    target_compile_options(single_tile_kernel_example PRIVATE
        -include ${KERNEL_HEADER}
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
    )
    
    if(hip_FOUND)
        target_link_libraries(single_tile_kernel_example PRIVATE hip::device hip::host)
    endif()
    
    message(STATUS "Building single_tile_kernel_example with real CK Tile kernel")
else()
    message(STATUS "Generated kernel not found - skipping single_tile_kernel_example")
    message(STATUS "  Generate with: cd codegen && python3 unified_gemm_codegen.py --preselected fp16_rcr_essential --output-dir ../build/generated_kernels --datatype fp16 --layout rcr")
endif()

