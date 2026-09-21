# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# CK Tile GEMM Unified Code Generator

cmake_minimum_required(VERSION 3.16)

# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Configuration
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/unified_gemm_codegen.py")
set(CODEGEN_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/default_config.json")
set(CODEGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated/tile_gemm")

# Configurable options
set(CK_TILE_GEMM_DATATYPE "fp16" CACHE STRING "GEMM data type (fp16, bf16, fp32, fp8, bf8, int8)")
set(CK_TILE_GEMM_LAYOUT "rcr" CACHE STRING "GEMM layout (rcr, rrr, crr, ccr)")
set(CK_TILE_GEMM_VARIANTS "standard" CACHE STRING "GEMM variants (standard, preshuffle, multi_d)")
set(CK_TILE_GEMM_GPU_TARGET "gfx942" CACHE STRING "Target GPU architecture")
set(CK_TILE_GEMM_PARALLEL ON CACHE BOOL "Enable parallel generation")

# Custom target to run code generation
add_custom_target(generate_tile_gemm_kernels
    COMMAND ${Python3_EXECUTABLE} ${CODEGEN_SCRIPT}
        --output-dir ${CODEGEN_OUTPUT_DIR}
        --datatype ${CK_TILE_GEMM_DATATYPE}
        --layout ${CK_TILE_GEMM_LAYOUT}
        --gpu-target ${CK_TILE_GEMM_GPU_TARGET}
        --config ${CODEGEN_CONFIG}
        --variants ${CK_TILE_GEMM_VARIANTS}
        $<$<NOT:$<BOOL:${CK_TILE_GEMM_PARALLEL}>>:--no-parallel>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating CK Tile GEMM kernels and dispatcher wrappers..."
    VERBATIM
)

# Create output directory
file(MAKE_DIRECTORY ${CODEGEN_OUTPUT_DIR})

# Add generated headers to include path
include_directories(${CODEGEN_OUTPUT_DIR})

# Installation
install(FILES
    ${CODEGEN_SCRIPT}
    ${CODEGEN_CONFIG}
    README.md
    DESTINATION share/ck_tile/codegen
)

# Helper function for projects to generate kernels
function(ck_tile_generate_gemm_kernels)
    set(options PARALLEL)
    set(oneValueArgs OUTPUT_DIR DATATYPE LAYOUT GPU_TARGET CONFIG)
    set(multiValueArgs VARIANTS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Set defaults
    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated/tile_gemm")
    endif()
    if(NOT ARG_DATATYPE)
        set(ARG_DATATYPE "fp16")
    endif()
    if(NOT ARG_LAYOUT)
        set(ARG_LAYOUT "rcr")
    endif()
    if(NOT ARG_GPU_TARGET)
        set(ARG_GPU_TARGET "gfx942")
    endif()
    if(NOT ARG_CONFIG)
        set(ARG_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/default_config.json")
    endif()
    if(NOT ARG_VARIANTS)
        set(ARG_VARIANTS "standard")
    endif()
    
    # Build command
    set(CMD ${Python3_EXECUTABLE} ${CODEGEN_SCRIPT}
        --output-dir ${ARG_OUTPUT_DIR}
        --datatype ${ARG_DATATYPE}
        --layout ${ARG_LAYOUT}
        --gpu-target ${ARG_GPU_TARGET}
        --config ${ARG_CONFIG}
        --variants ${ARG_VARIANTS}
    )
    
    if(NOT ARG_PARALLEL)
        list(APPEND CMD --no-parallel)
    endif()
    
    # Execute
    execute_process(
        COMMAND ${CMD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE OUTPUT
        ERROR_VARIABLE ERROR
    )
    
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate GEMM kernels:\n${ERROR}")
    else()
        message(STATUS "Generated GEMM kernels: ${OUTPUT}")
    endif()
endfunction()

# Example usage documentation
message(STATUS "CK Tile GEMM Code Generator configured")
message(STATUS "  Script: ${CODEGEN_SCRIPT}")
message(STATUS "  Config: ${CODEGEN_CONFIG}")
message(STATUS "  Output: ${CODEGEN_OUTPUT_DIR}")
message(STATUS "")
message(STATUS "To generate kernels:")
message(STATUS "  cmake --build . --target generate_tile_gemm_kernels")
message(STATUS "")
message(STATUS "Or use CMake function:")
message(STATUS "  ck_tile_generate_gemm_kernels(")
message(STATUS "    OUTPUT_DIR ./generated")
message(STATUS "    DATATYPE fp16")
message(STATUS "    LAYOUT rcr")
message(STATUS "    VARIANTS standard preshuffle multi_d")
message(STATUS "    PARALLEL")
message(STATUS "  )")
