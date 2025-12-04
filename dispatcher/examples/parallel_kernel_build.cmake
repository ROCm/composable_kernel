# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# =============================================================================
# Parallel Kernel Compilation Support
# =============================================================================
#
# This module provides functions for parallel compilation of individual kernels.
# Each kernel is compiled as a separate OBJECT library, then linked into the
# final shared library. This enables maximum parallelism with make -j.
#
# Usage:
#   include(parallel_kernel_build.cmake)
#   
#   # Generate kernel wrapper sources
#   generate_kernel_wrappers(
#       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_wrappers
#       KERNEL_HEADERS ${GENERATED_KERNEL_HEADERS}
#       OUTPUT_SOURCES WRAPPER_SOURCES
#   )
#   
#   # Build parallel kernel objects
#   build_parallel_kernels(
#       TARGET_NAME my_parallel_kernels
#       SOURCES ${WRAPPER_SOURCES}
#       INCLUDE_DIRS ${KERNEL_INCLUDE_DIRS}
#   )
#   
#   # Link into final library
#   target_link_libraries(my_lib PRIVATE my_parallel_kernels)
#
# =============================================================================

include_guard()

# Global counter for unique target names
set(_PARALLEL_KERNEL_COUNTER 0 CACHE INTERNAL "Parallel kernel counter")

# =============================================================================
# generate_kernel_wrapper_source
# Creates a .cpp wrapper file for a kernel header
# =============================================================================
function(generate_kernel_wrapper_source KERNEL_HEADER OUTPUT_DIR OUTPUT_VAR)
    get_filename_component(kernel_name ${KERNEL_HEADER} NAME_WE)
    set(wrapper_file "${OUTPUT_DIR}/${kernel_name}_wrapper.cpp")
    
    file(WRITE ${wrapper_file}
"// Auto-generated kernel wrapper for parallel compilation
// Kernel: ${kernel_name}

#include \"${KERNEL_HEADER}\"

// Force instantiation of kernel templates
namespace {
    // The kernel is instantiated via -include flag
    // This file exists to create a separate compilation unit
    volatile int _${kernel_name}_dummy = 0;
}
")
    
    set(${OUTPUT_VAR} ${wrapper_file} PARENT_SCOPE)
endfunction()

# =============================================================================
# generate_kernel_wrappers
# Generates wrapper sources for all kernel headers
# =============================================================================
function(generate_kernel_wrappers)
    cmake_parse_arguments(GKW "" "OUTPUT_DIR" "KERNEL_HEADERS;OUTPUT_SOURCES" ${ARGN})
    
    if(NOT GKW_OUTPUT_DIR)
        message(FATAL_ERROR "generate_kernel_wrappers: OUTPUT_DIR is required")
    endif()
    
    file(MAKE_DIRECTORY ${GKW_OUTPUT_DIR})
    
    set(wrapper_sources "")
    
    foreach(header ${GKW_KERNEL_HEADERS})
        generate_kernel_wrapper_source(${header} ${GKW_OUTPUT_DIR} wrapper)
        list(APPEND wrapper_sources ${wrapper})
    endforeach()
    
    if(GKW_OUTPUT_SOURCES)
        set(${GKW_OUTPUT_SOURCES} ${wrapper_sources} PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# add_kernel_object
# Creates an OBJECT library for a single kernel
# =============================================================================
function(add_kernel_object KERNEL_HEADER TARGET_PREFIX)
    cmake_parse_arguments(AKO "" "" "INCLUDE_DIRS;COMPILE_OPTIONS" ${ARGN})
    
    math(EXPR _PARALLEL_KERNEL_COUNTER "${_PARALLEL_KERNEL_COUNTER} + 1")
    set(_PARALLEL_KERNEL_COUNTER ${_PARALLEL_KERNEL_COUNTER} CACHE INTERNAL "")
    
    get_filename_component(kernel_name ${KERNEL_HEADER} NAME_WE)
    set(target_name "${TARGET_PREFIX}_${kernel_name}")
    
    # Create a minimal source file that includes the kernel
    set(wrapper_dir "${CMAKE_CURRENT_BINARY_DIR}/kernel_objects")
    file(MAKE_DIRECTORY ${wrapper_dir})
    
    set(wrapper_file "${wrapper_dir}/${kernel_name}_obj.cpp")
    file(WRITE ${wrapper_file}
"// Kernel object: ${kernel_name}
// This file is compiled with -include ${KERNEL_HEADER}
namespace { volatile int _ko_${_PARALLEL_KERNEL_COUNTER} = 0; }
")
    
    add_library(${target_name} OBJECT ${wrapper_file})
    
    if(AKO_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${AKO_INCLUDE_DIRS})
    endif()
    
    target_compile_options(${target_name} PRIVATE
        -include ${KERNEL_HEADER}
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        ${AKO_COMPILE_OPTIONS}
    )
    
    if(hip_FOUND)
        target_link_libraries(${target_name} PRIVATE hip::device hip::host)
    endif()
    
    # Return the target name
    set(KERNEL_OBJECT_TARGET ${target_name} PARENT_SCOPE)
endfunction()

# =============================================================================
# build_parallel_kernels
# Creates OBJECT libraries for multiple kernels that can compile in parallel
# =============================================================================
function(build_parallel_kernels)
    cmake_parse_arguments(BPK "" "TARGET_NAME;OUTPUT_DIR" 
        "KERNEL_HEADERS;INCLUDE_DIRS;COMPILE_OPTIONS;DEPENDENCIES" ${ARGN})
    
    if(NOT BPK_TARGET_NAME)
        message(FATAL_ERROR "build_parallel_kernels: TARGET_NAME is required")
    endif()
    
    if(NOT BPK_OUTPUT_DIR)
        set(BPK_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/parallel_kernels/${BPK_TARGET_NAME}")
    endif()
    
    file(MAKE_DIRECTORY ${BPK_OUTPUT_DIR})
    
    set(object_targets "")
    
    foreach(header ${BPK_KERNEL_HEADERS})
        add_kernel_object(${header} ${BPK_TARGET_NAME}
            INCLUDE_DIRS ${BPK_INCLUDE_DIRS}
            COMPILE_OPTIONS ${BPK_COMPILE_OPTIONS}
        )
        list(APPEND object_targets ${KERNEL_OBJECT_TARGET})
        
        # Add dependencies
        if(BPK_DEPENDENCIES)
            add_dependencies(${KERNEL_OBJECT_TARGET} ${BPK_DEPENDENCIES})
        endif()
    endforeach()
    
    # Create an interface library that aggregates all objects
    add_library(${BPK_TARGET_NAME} INTERFACE)
    target_sources(${BPK_TARGET_NAME} INTERFACE
        $<TARGET_OBJECTS:${object_targets}>
    )
    
    # Store the list of object targets for reference
    set_property(TARGET ${BPK_TARGET_NAME} PROPERTY KERNEL_OBJECTS "${object_targets}")
    
    message(STATUS "Created parallel kernel target ${BPK_TARGET_NAME} with ${list_length} kernels")
endfunction()

# =============================================================================
# Example Usage
# =============================================================================
#
# # Find all generated kernel headers
# file(GLOB KERNEL_HEADERS "${KERNEL_OUTPUT_DIR}/*.hpp")
#
# # Build parallel kernel objects
# build_parallel_kernels(
#     TARGET_NAME gemm_kernels_parallel
#     KERNEL_HEADERS ${KERNEL_HEADERS}
#     INCLUDE_DIRS 
#         ${CMAKE_SOURCE_DIR}/include
#         ${DISPATCHER_INCLUDE_DIR}
#     COMPILE_OPTIONS
#         -DGEMM_KERNEL_AVAILABLE=1
#     DEPENDENCIES
#         generate_gemm_kernels
# )
#
# # Create the shared library using the parallel-compiled kernels
# add_library(dispatcher_gemm_lib SHARED
#     ${CMAKE_SOURCE_DIR}/bindings/ctypes/gemm_ctypes_lib.cpp
# )
# target_link_libraries(dispatcher_gemm_lib PRIVATE
#     gemm_kernels_parallel
#     ck_tile_dispatcher
# )
#
# =============================================================================

