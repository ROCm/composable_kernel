# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# CK Tile Dispatcher - ctypes Bindings
# =============================================================================
#
# Provides shared libraries with C API for Python ctypes integration.
#
# Targets:
#   - dispatcher_gemm_lib      : GEMM dispatcher library
#   - dispatcher_conv_lib      : Convolution dispatcher library (forward + bwd_data)
#   - dispatcher_conv_bwdw_lib : Convolution backward weight library
#   - gpu_helper               : GPU helper executable for Python
#

cmake_minimum_required(VERSION 3.16)

# Helper function to add a ctypes library
function(add_ctypes_library TARGET_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "CONV" "KERNEL_HEADER" "" ${ARGN})
    
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})
    
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    
    target_link_libraries(${TARGET_NAME} PRIVATE
        hip::device
    )
    
    # Force-include kernel header if provided
    if(ARG_KERNEL_HEADER AND EXISTS ${ARG_KERNEL_HEADER})
        target_compile_options(${TARGET_NAME} PRIVATE
            -include ${ARG_KERNEL_HEADER}
        )
        if(ARG_CONV)
            target_compile_definitions(${TARGET_NAME} PRIVATE CONV_KERNEL_AVAILABLE)
        endif()
    endif()
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endfunction()

# =============================================================================
# GEMM ctypes Library
# =============================================================================

# Find a generated GEMM kernel header for the library
file(GLOB GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*.hpp")
if(GEMM_KERNEL_HEADERS)
    list(GET GEMM_KERNEL_HEADERS 0 GEMM_KERNEL_HEADER)
    message(STATUS "Found GEMM kernel for ctypes lib: ${GEMM_KERNEL_HEADER}")
    
    add_ctypes_library(dispatcher_gemm_lib 
        gemm_ctypes_lib.cpp 
        KERNEL_HEADER ${GEMM_KERNEL_HEADER}
    )
else()
    message(STATUS "No GEMM kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_gemm_lib SHARED gemm_ctypes_lib.cpp)
    target_include_directories(dispatcher_gemm_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_gemm_lib PRIVATE hip::device)
endif()

# =============================================================================
# Convolution ctypes Library (supports forward + bwd_data)
# =============================================================================

# Look for forward kernels
file(GLOB CONV_FWD_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_fwd_*.hpp")
# Look for backward data kernels  
file(GLOB CONV_BWDD_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_bwd_data_*.hpp")
# Fallback: any conv kernel (for backwards compatibility)
file(GLOB CONV_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_*.hpp")

add_library(dispatcher_conv_lib SHARED conv_ctypes_lib.cpp)
target_include_directories(dispatcher_conv_lib PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dispatcher/include
)
target_link_libraries(dispatcher_conv_lib PRIVATE hip::device)
set_target_properties(dispatcher_conv_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)

# Add forward kernel if available
if(CONV_FWD_KERNEL_HEADERS)
    list(GET CONV_FWD_KERNEL_HEADERS 0 CONV_FWD_KERNEL_HEADER)
    message(STATUS "Found Conv FWD kernel for ctypes lib: ${CONV_FWD_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_FWD_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_KERNEL_AVAILABLE)
elseif(CONV_KERNEL_HEADERS)
    # Fallback to any conv kernel
    list(GET CONV_KERNEL_HEADERS 0 CONV_KERNEL_HEADER)
    message(STATUS "Found Conv kernel for ctypes lib: ${CONV_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_KERNEL_AVAILABLE)
else()
    message(STATUS "No Conv FWD kernel found for ctypes lib - building without kernel")
endif()

# Add backward data kernel if available
if(CONV_BWDD_KERNEL_HEADERS)
    list(GET CONV_BWDD_KERNEL_HEADERS 0 CONV_BWDD_KERNEL_HEADER)
    message(STATUS "Found Conv BWD_DATA kernel for ctypes lib: ${CONV_BWD_DATA_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_BWDD_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_BWD_DATA_AVAILABLE)
endif()

# =============================================================================
# Convolution Backward Weight ctypes Library (separate lib for bwd_weight)
# =============================================================================

file(GLOB CONV_BWDW_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_*bwd_weight*.hpp")
if(CONV_BWDW_KERNEL_HEADERS)
    list(GET CONV_BWDW_KERNEL_HEADERS 0 CONV_BWDW_KERNEL_HEADER)
    message(STATUS "Found Conv BwdWeight kernel for ctypes lib: ${CONV_BWDW_KERNEL_HEADER}")
    
    add_library(dispatcher_conv_bwdw_lib SHARED conv_bwdw_ctypes_lib.cpp)
    target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device)
    target_compile_options(dispatcher_conv_bwdw_lib PRIVATE
        -include ${CONV_BWDW_KERNEL_HEADER}
    )
    target_compile_definitions(dispatcher_conv_bwdw_lib PRIVATE CONV_BWD_WEIGHT_AVAILABLE)
    set_target_properties(dispatcher_conv_bwdw_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
else()
    message(STATUS "No Conv BwdWeight kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_conv_bwdw_lib SHARED conv_bwdw_ctypes_lib.cpp)
    target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device)
    set_target_properties(dispatcher_conv_bwdw_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

# =============================================================================
# GPU Helper Executable
# =============================================================================

if(GEMM_KERNEL_HEADERS)
    add_executable(gpu_helper gpu_helper.cpp)
    
    target_include_directories(gpu_helper PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    
    target_link_libraries(gpu_helper PRIVATE
        hip::device
    )
    
    target_compile_options(gpu_helper PRIVATE
        -include ${GEMM_KERNEL_HEADER}
    )
    
    set_target_properties(gpu_helper PROPERTIES
        CXX_STANDARD 17
    )
endif()

