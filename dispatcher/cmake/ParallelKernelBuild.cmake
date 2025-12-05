# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# =============================================================================
# Parallel Kernel Build Module
# =============================================================================
#
# This module provides functions to compile each kernel as a separate object
# file, enabling maximum parallelism with verbose progress output.
#
# Usage:
#   include(cmake/ParallelKernelBuild.cmake)
#   
#   # Option 1: Build from generated kernel directory
#   build_kernel_library_from_dir(
#       NAME my_kernels
#       KERNEL_DIR ${CMAKE_BINARY_DIR}/generated_kernels
#       PATTERN "gemm_fp16_*.hpp"
#   )
#   
#   # Option 2: Build from explicit list
#   build_kernel_library(
#       NAME my_kernels
#       KERNELS kernel1.hpp kernel2.hpp kernel3.hpp
#   )
#
# The resulting library (libmy_kernels.so) can be linked to executables.
# During build, you'll see:
#   [  5%] Building kernel: gemm_fp16_rcr_128x128x32.hpp
#   [ 10%] Building kernel: gemm_fp16_rcr_256x256x64.hpp
#   ...
#
# =============================================================================

# Generate a wrapper .cpp file for a kernel header
# The wrapper instantiates the kernel so it gets compiled
function(generate_kernel_wrapper KERNEL_HPP OUTPUT_CPP KERNEL_TYPE)
    get_filename_component(KERNEL_NAME ${KERNEL_HPP} NAME_WE)
    
    # Generate wrapper content based on kernel type
    if(KERNEL_TYPE STREQUAL "gemm")
        set(WRAPPER_CONTENT
"// Auto-generated wrapper for ${KERNEL_NAME}
#include \"${KERNEL_HPP}\"

// Explicit template instantiation marker
namespace ck_tile { namespace dispatcher { namespace generated {
    // Force symbol emission for kernel registration
    volatile bool _kernel_${KERNEL_NAME}_registered = true;
}}}
")
    elseif(KERNEL_TYPE STREQUAL "conv")
        set(WRAPPER_CONTENT
"// Auto-generated wrapper for ${KERNEL_NAME}
#include \"${KERNEL_HPP}\"

namespace ck_tile { namespace dispatcher { namespace generated {
    volatile bool _kernel_${KERNEL_NAME}_registered = true;
}}}
")
    else()
        set(WRAPPER_CONTENT
"// Auto-generated wrapper for ${KERNEL_NAME}
#include \"${KERNEL_HPP}\"
")
    endif()
    
    file(WRITE ${OUTPUT_CPP} "${WRAPPER_CONTENT}")
endfunction()


# Build a kernel library from a directory of kernel headers
# Each kernel is compiled as a separate object for parallel builds
function(build_kernel_library_from_dir)
    cmake_parse_arguments(ARG "" "NAME;KERNEL_DIR;PATTERN;TYPE" "" ${ARGN})
    
    if(NOT ARG_NAME)
        message(FATAL_ERROR "build_kernel_library_from_dir: NAME is required")
    endif()
    if(NOT ARG_KERNEL_DIR)
        message(FATAL_ERROR "build_kernel_library_from_dir: KERNEL_DIR is required")
    endif()
    if(NOT ARG_PATTERN)
        set(ARG_PATTERN "*.hpp")
    endif()
    if(NOT ARG_TYPE)
        set(ARG_TYPE "gemm")
    endif()
    
    # Find all kernel headers
    file(GLOB KERNEL_HEADERS "${ARG_KERNEL_DIR}/${ARG_PATTERN}")
    
    if(NOT KERNEL_HEADERS)
        message(WARNING "No kernel headers found matching ${ARG_KERNEL_DIR}/${ARG_PATTERN}")
        return()
    endif()
    
    list(LENGTH KERNEL_HEADERS NUM_KERNELS)
    message(STATUS "Found ${NUM_KERNELS} kernels for library ${ARG_NAME}")
    
    # Create wrapper directory
    set(WRAPPER_DIR "${CMAKE_BINARY_DIR}/kernel_wrappers/${ARG_NAME}")
    file(MAKE_DIRECTORY ${WRAPPER_DIR})
    
    # Generate wrappers and collect sources
    set(KERNEL_SOURCES "")
    foreach(KERNEL_HPP ${KERNEL_HEADERS})
        get_filename_component(KERNEL_NAME ${KERNEL_HPP} NAME_WE)
        set(WRAPPER_CPP "${WRAPPER_DIR}/${KERNEL_NAME}.cpp")
        
        generate_kernel_wrapper(${KERNEL_HPP} ${WRAPPER_CPP} ${ARG_TYPE})
        list(APPEND KERNEL_SOURCES ${WRAPPER_CPP})
    endforeach()
    
    # Create shared library from all kernel objects
    add_library(${ARG_NAME} SHARED ${KERNEL_SOURCES})
    
    target_include_directories(${ARG_NAME} PRIVATE
        ${ARG_KERNEL_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/../include
    )
    
    target_compile_options(${ARG_NAME} PRIVATE
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
    )
    
    if(hip_FOUND)
        target_link_libraries(${ARG_NAME} PRIVATE hip::device hip::host)
    endif()
    
    set_target_properties(${ARG_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()


# Build kernel library with verbose per-kernel progress
# This creates individual targets for maximum visibility
function(build_kernel_library_verbose)
    cmake_parse_arguments(ARG "" "NAME;KERNEL_DIR;PATTERN;TYPE" "" ${ARGN})
    
    if(NOT ARG_NAME)
        message(FATAL_ERROR "build_kernel_library_verbose: NAME is required")
    endif()
    if(NOT ARG_KERNEL_DIR)
        message(FATAL_ERROR "build_kernel_library_verbose: KERNEL_DIR is required")
    endif()
    if(NOT ARG_PATTERN)
        set(ARG_PATTERN "*.hpp")
    endif()
    if(NOT ARG_TYPE)
        set(ARG_TYPE "gemm")
    endif()
    
    # Find all kernel headers
    file(GLOB KERNEL_HEADERS "${ARG_KERNEL_DIR}/${ARG_PATTERN}")
    
    if(NOT KERNEL_HEADERS)
        message(WARNING "No kernel headers found matching ${ARG_KERNEL_DIR}/${ARG_PATTERN}")
        return()
    endif()
    
    list(LENGTH KERNEL_HEADERS NUM_KERNELS)
    message(STATUS "Building ${NUM_KERNELS} kernels as individual objects for ${ARG_NAME}")
    
    # Create wrapper directory
    set(WRAPPER_DIR "${CMAKE_BINARY_DIR}/kernel_wrappers/${ARG_NAME}")
    file(MAKE_DIRECTORY ${WRAPPER_DIR})
    
    # Create object library for each kernel (enables parallel compilation)
    set(KERNEL_OBJECTS "")
    set(KERNEL_IDX 0)
    foreach(KERNEL_HPP ${KERNEL_HEADERS})
        math(EXPR KERNEL_IDX "${KERNEL_IDX} + 1")
        get_filename_component(KERNEL_NAME ${KERNEL_HPP} NAME_WE)
        set(WRAPPER_CPP "${WRAPPER_DIR}/${KERNEL_NAME}.cpp")
        set(OBJ_TARGET "${ARG_NAME}_obj_${KERNEL_NAME}")
        
        # Generate wrapper
        generate_kernel_wrapper(${KERNEL_HPP} ${WRAPPER_CPP} ${ARG_TYPE})
        
        # Create OBJECT library for this kernel
        add_library(${OBJ_TARGET} OBJECT ${WRAPPER_CPP})
        
        target_include_directories(${OBJ_TARGET} PRIVATE
            ${ARG_KERNEL_DIR}
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/../include
        )
        
        target_compile_options(${OBJ_TARGET} PRIVATE
            -mllvm -enable-noalias-to-md-conversion=0
            -Wno-undefined-func-template
            -Wno-float-equal
            --offload-compress
        )
        
        if(hip_FOUND)
            target_link_libraries(${OBJ_TARGET} PRIVATE hip::device hip::host)
        endif()
        
        set_target_properties(${OBJ_TARGET} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
        )
        
        # Add custom message for verbose output
        add_custom_command(TARGET ${OBJ_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "[${KERNEL_IDX}/${NUM_KERNELS}] Building kernel: ${KERNEL_NAME}"
            VERBATIM
        )
        
        list(APPEND KERNEL_OBJECTS $<TARGET_OBJECTS:${OBJ_TARGET}>)
    endforeach()
    
    # Link all object files into shared library
    add_library(${ARG_NAME} SHARED ${KERNEL_OBJECTS})
    
    if(hip_FOUND)
        target_link_libraries(${ARG_NAME} PRIVATE hip::device hip::host)
    endif()
    
    set_target_properties(${ARG_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "Library ${ARG_NAME} will be built from ${NUM_KERNELS} parallel kernel objects")
endfunction()


# Simple function to add a kernel object target with verbose name
function(add_kernel_object KERNEL_HPP KERNEL_DIR OBJ_LIST_VAR INDEX TOTAL)
    get_filename_component(KERNEL_NAME ${KERNEL_HPP} NAME_WE)
    set(OBJ_TARGET "kernel_${KERNEL_NAME}")
    
    # Create a minimal .cpp that includes the kernel
    set(WRAPPER_CPP "${CMAKE_BINARY_DIR}/kernel_wrappers/${KERNEL_NAME}.cpp")
    file(WRITE ${WRAPPER_CPP} 
"// Kernel wrapper: ${KERNEL_NAME}
#include \"${KERNEL_HPP}\"
namespace { volatile bool _k = true; }
")
    
    add_library(${OBJ_TARGET} OBJECT ${WRAPPER_CPP})
    
    target_include_directories(${OBJ_TARGET} PRIVATE ${KERNEL_DIR})
    
    target_compile_options(${OBJ_TARGET} PRIVATE
        -mllvm -enable-noalias-to-md-conversion=0
        -Wno-undefined-func-template
        --offload-compress
    )
    
    if(hip_FOUND)
        target_link_libraries(${OBJ_TARGET} PRIVATE hip::device hip::host)
    endif()
    
    set_target_properties(${OBJ_TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Append to output list
    set(${OBJ_LIST_VAR} ${${OBJ_LIST_VAR}} $<TARGET_OBJECTS:${OBJ_TARGET}> PARENT_SCOPE)
endfunction()

