# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Find generated kernel header for force-include
file(GLOB KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels/gemm_fp16_rcr_compv4*128x128x32*.hpp")
if(KERNEL_HEADERS)
    list(GET KERNEL_HEADERS 0 KERNEL_HEADER)
else()
    set(KERNEL_HEADER "")
endif()

# Helper function to add a GPU example with force-included kernel
function(add_gpu_example NAME SOURCE)
    add_executable(${NAME} cpp/${SOURCE})
    
    target_link_libraries(${NAME} PRIVATE ck_tile_dispatcher)
    
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include              # CK root include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include                 # Dispatcher include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels # Generated kernels
    )
    
    target_compile_options(${NAME} PRIVATE
        -include ${KERNEL_HEADER}
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    
    if(hip_FOUND)
        target_link_libraries(${NAME} PRIVATE hip::device hip::host)
    endif()
endfunction()

if(KERNEL_HEADER AND EXISTS "${KERNEL_HEADER}")
    message(STATUS "Building examples with generated kernel: ${KERNEL_HEADER}")
    
    # Numbered examples (ordered by complexity)
    add_gpu_example(example_01_basic_gemm       01_basic_gemm.cpp)
    add_gpu_example(example_02_multi_size       02_multi_size.cpp)
    add_gpu_example(example_03_benchmark        03_benchmark.cpp)
    add_gpu_example(example_04_validation       04_validation.cpp)
    add_gpu_example(example_05_heuristics       05_heuristics.cpp)
    add_gpu_example(example_06_json_export      06_json_export.cpp)
    add_gpu_example(example_07_preshuffle       07_preshuffle.cpp)
    add_gpu_example(example_08_multi_d          08_multi_d.cpp)
    add_gpu_example(example_09_multi_registry   09_multi_registry.cpp)
    
    # Python utilities
    add_gpu_example(python_gpu_helper           python_gpu_helper.cpp)
    
    # Dynamic library for Python ctypes
    add_library(dispatcher_gemm SHARED cpp/dispatcher_dynamic_lib.cpp)
    target_link_libraries(dispatcher_gemm PRIVATE ck_tile_dispatcher)
    target_include_directories(dispatcher_gemm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/generated_kernels
    )
    target_compile_options(dispatcher_gemm PRIVATE
        -include ${KERNEL_HEADER}
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    if(hip_FOUND)
        target_link_libraries(dispatcher_gemm PRIVATE hip::device hip::host)
    endif()
    
    message(STATUS "Built examples: example_01 through example_09, plus utilities")
else()
    message(STATUS "Generated kernels not found - skipping GPU examples")
    message(STATUS "  Generate with: cd codegen && python3 unified_gemm_codegen.py --preselected fp16_rcr_essential")
endif()

message(STATUS "Examples configuration complete")
