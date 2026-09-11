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
#   - dispatcher_mx_gemm_lib   : MX-GEMM (microscaling) dispatcher library (gfx950)
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
# MX-GEMM ctypes Library (microscaling GEMM, gfx950/MI350 only)
# =============================================================================
#
# Force-includes a generated mx_gemm kernel header (SelectedKernel/KERNEL_NAME/
# MxGemmHostArgs/ScaleType/...). The lib also includes the Old-TE common helper
# (common/utils.hpp) for the free-function is_row_major(Layout), so add the
# tile_engine/ops include roots.

set(MX_GEMM_TE_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/tile_engine/ops
    ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm
    ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/mx_gemm
)

# Resolve the configured GPU target GENERICALLY -- do NOT hardcode/default the
# arch to gfx950. mx_gemm is gfx950-only (it calls the gfx950-only
# preShuffleScaleBuffer_gfx950 helper), and that requirement is enforced IN THE
# SOURCE: mx_gemm_ctypes_lib.cpp #errors if GFX_ARCH is unset and static_asserts
# GFX_ARCH == "gfx950" (plus a runtime device guard). Here we simply pass the
# real build target through as -DGFX_ARCH and only wire up the target when the
# build actually targets gfx950, so a gfx942/other dispatcher build is neither
# broken by the static_assert nor silently built for an unsupported arch.
if(DEFINED GPU_TARGETS AND NOT GPU_TARGETS STREQUAL "")
    string(REPLACE ";" " " _mx_gpu_targets_space "${GPU_TARGETS}")
    string(REPLACE " " ";" _mx_gpu_targets_list "${_mx_gpu_targets_space}")
    list(GET _mx_gpu_targets_list 0 MX_GEMM_GPU_TARGET)
elseif(DEFINED CK_TILE_GEMM_GPU_TARGET AND NOT CK_TILE_GEMM_GPU_TARGET STREQUAL "")
    set(MX_GEMM_GPU_TARGET "${CK_TILE_GEMM_GPU_TARGET}")
else()
    set(MX_GEMM_GPU_TARGET "")
endif()

if(NOT MX_GEMM_GPU_TARGET STREQUAL "gfx950")
    message(STATUS
        "MX-GEMM ctypes lib is gfx950-only; skipping for GPU target "
        "'${MX_GEMM_GPU_TARGET}' (configure with GPU_TARGETS=gfx950 to enable).")
else()
    file(GLOB MX_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/mx_gemm_*.hpp")
    if(MX_GEMM_KERNEL_HEADERS)
        list(GET MX_GEMM_KERNEL_HEADERS 0 MX_GEMM_KERNEL_HEADER)
        message(STATUS "Found MX-GEMM kernel for ctypes lib: ${MX_GEMM_KERNEL_HEADER}")

        add_ctypes_library(dispatcher_mx_gemm_lib
            mx_gemm_ctypes_lib.cpp
            KERNEL_HEADER ${MX_GEMM_KERNEL_HEADER}
        )
        target_include_directories(dispatcher_mx_gemm_lib PRIVATE ${MX_GEMM_TE_INCLUDE_DIRS})
        # The generated header only exports SelectedKernel/KERNEL_NAME/ScaleType/...
        # under this define. Pass the resolved build target through as -DGFX_ARCH
        # (the source #errors if unset and static_asserts it is gfx950).
        target_compile_definitions(dispatcher_mx_gemm_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE
            GFX_ARCH="${MX_GEMM_GPU_TARGET}"
        )
    else()
        message(STATUS "No MX-GEMM kernel found for ctypes lib - building without kernel")
        add_library(dispatcher_mx_gemm_lib SHARED mx_gemm_ctypes_lib.cpp)
        target_include_directories(dispatcher_mx_gemm_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
            ${MX_GEMM_TE_INCLUDE_DIRS}
        )
        target_link_libraries(dispatcher_mx_gemm_lib PRIVATE hip::device)
        # Pass the resolved build target through as -DGFX_ARCH (the source #errors
        # if unset and static_asserts it is gfx950).
        target_compile_definitions(dispatcher_mx_gemm_lib PRIVATE
            GFX_ARCH="${MX_GEMM_GPU_TARGET}")
        set_target_properties(dispatcher_mx_gemm_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    endif()
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
# GroupedGemm BQuant ctypes Library
# =============================================================================

file(GLOB BQUANT_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_bquant_*.hpp")
if(BQUANT_GEMM_KERNEL_HEADERS)
    list(GET BQUANT_GEMM_KERNEL_HEADERS 0 BQUANT_GEMM_KERNEL_HEADER)
    message(STATUS "Found BQuant GEMM kernel for ctypes lib: ${BQUANT_GEMM_KERNEL_HEADER}")

    add_ctypes_library(dispatcher_grouped_gemm_bquant_lib
        grouped_gemm_bquant_ctypes_lib.cpp
        KERNEL_HEADER ${BQUANT_GEMM_KERNEL_HEADER}
    )
    # Resolve a single GFX arch string from CMAKE_HIP_ARCHITECTURES if available,
    # falling back to the cache variable CK_TILE_BQUANT_GFX_ARCH (default gfx942).
    # The Python build path always passes -DGFX_ARCH=... explicitly via hipcc, so
    # this default only affects builds driven through CMake directly.
    if(NOT DEFINED CK_TILE_BQUANT_GFX_ARCH OR CK_TILE_BQUANT_GFX_ARCH STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _bquant_gfx_arch)
        else()
            set(_bquant_gfx_arch "gfx942")
        endif()
        set(CK_TILE_BQUANT_GFX_ARCH "${_bquant_gfx_arch}" CACHE STRING
            "GFX arch for the CMake-built BQuant ctypes .so (default: first CMAKE_HIP_ARCHITECTURES or gfx942)")
    endif()
    message(STATUS "BQuant ctypes lib GFX_ARCH: ${CK_TILE_BQUANT_GFX_ARCH}")

    target_compile_definitions(dispatcher_grouped_gemm_bquant_lib PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${CK_TILE_BQUANT_GFX_ARCH}"
    )
else()
    message(STATUS "No BQuant GEMM kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_grouped_gemm_bquant_lib SHARED grouped_gemm_bquant_ctypes_lib.cpp)
    target_include_directories(dispatcher_grouped_gemm_bquant_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_grouped_gemm_bquant_lib PRIVATE hip::device)
    set_target_properties(dispatcher_grouped_gemm_bquant_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

# =============================================================================
# GroupedGemm AQuant ctypes Library
# =============================================================================

file(GLOB AQUANT_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_aquant_*.hpp")
if(AQUANT_GEMM_KERNEL_HEADERS)
    list(GET AQUANT_GEMM_KERNEL_HEADERS 0 AQUANT_GEMM_KERNEL_HEADER)
    message(STATUS "Found AQuant GEMM kernel for ctypes lib: ${AQUANT_GEMM_KERNEL_HEADER}")

    add_ctypes_library(dispatcher_grouped_gemm_aquant_lib
        grouped_gemm_aquant_ctypes_lib.cpp
        KERNEL_HEADER ${AQUANT_GEMM_KERNEL_HEADER}
    )
    if(NOT DEFINED CK_TILE_AQUANT_GFX_ARCH OR CK_TILE_AQUANT_GFX_ARCH STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _aquant_gfx_arch)
        else()
            set(_aquant_gfx_arch "gfx942")
        endif()
        set(CK_TILE_AQUANT_GFX_ARCH "${_aquant_gfx_arch}" CACHE STRING
            "GFX arch for the CMake-built AQuant ctypes .so (default: first CMAKE_HIP_ARCHITECTURES or gfx942)")
    endif()
    message(STATUS "AQuant ctypes lib GFX_ARCH: ${CK_TILE_AQUANT_GFX_ARCH}")
    target_compile_definitions(dispatcher_grouped_gemm_aquant_lib PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${CK_TILE_AQUANT_GFX_ARCH}"
    )
else()
    message(STATUS "No AQuant GEMM kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_grouped_gemm_aquant_lib SHARED grouped_gemm_aquant_ctypes_lib.cpp)
    target_include_directories(dispatcher_grouped_gemm_aquant_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_grouped_gemm_aquant_lib PRIVATE hip::device)
    set_target_properties(dispatcher_grouped_gemm_aquant_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

# =============================================================================
# GroupedGemm ABQuant ctypes Library
# =============================================================================

file(GLOB ABQUANT_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_abquant_*.hpp")
if(ABQUANT_GEMM_KERNEL_HEADERS)
    list(GET ABQUANT_GEMM_KERNEL_HEADERS 0 ABQUANT_GEMM_KERNEL_HEADER)
    message(STATUS "Found ABQuant GEMM kernel for ctypes lib: ${ABQUANT_GEMM_KERNEL_HEADER}")

    add_ctypes_library(dispatcher_grouped_gemm_abquant_lib
        grouped_gemm_abquant_ctypes_lib.cpp
        KERNEL_HEADER ${ABQUANT_GEMM_KERNEL_HEADER}
    )
    if(NOT DEFINED CK_TILE_ABQUANT_GFX_ARCH OR CK_TILE_ABQUANT_GFX_ARCH STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _abquant_gfx_arch)
        else()
            set(_abquant_gfx_arch "gfx942")
        endif()
        set(CK_TILE_ABQUANT_GFX_ARCH "${_abquant_gfx_arch}" CACHE STRING
            "GFX arch for the CMake-built ABQuant ctypes .so (default: first CMAKE_HIP_ARCHITECTURES or gfx942)")
    endif()
    message(STATUS "ABQuant ctypes lib GFX_ARCH: ${CK_TILE_ABQUANT_GFX_ARCH}")
    target_compile_definitions(dispatcher_grouped_gemm_abquant_lib PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${CK_TILE_ABQUANT_GFX_ARCH}"
    )
else()
    message(STATUS "No ABQuant GEMM kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_grouped_gemm_abquant_lib SHARED grouped_gemm_abquant_ctypes_lib.cpp)
    target_include_directories(dispatcher_grouped_gemm_abquant_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_grouped_gemm_abquant_lib PRIVATE hip::device)
    set_target_properties(dispatcher_grouped_gemm_abquant_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

# =============================================================================
# TileEngine -> Dispatcher Bridge ctypes Libraries (append-only shared block)
# =============================================================================
#
# The bridge PRs (#9305 gemm_multi_abd, #9306 batched_gemm, #9328
# batched_contraction) each add a registry-bypass ctypes .so target. To keep the
# branches mutually conflict-proof, this block is BYTE-IDENTICAL on every branch
# that touches this file. Each target is doubly guarded:
#   (1) on its generated kernel-header glob, AND
#   (2) on if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/<op>_ctypes_lib.cpp)
# so a branch that does not ship a given <op>_ctypes_lib.cpp simply skips that
# target at configure time (no error). The Python bridges build these .so files
# at runtime via hipcc; these CMake targets exist for CMake-driven / CI builds.

# --- GEMM Multi-ABD (registry-bypass, array-pointer ABI) ---------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_multi_abd_ctypes_lib.cpp)
    file(GLOB GEMM_MULTI_ABD_KERNEL_HEADERS
        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*_multiabd_*.hpp")
    if(GEMM_MULTI_ABD_KERNEL_HEADERS)
        list(SORT GEMM_MULTI_ABD_KERNEL_HEADERS)
        list(GET GEMM_MULTI_ABD_KERNEL_HEADERS 0 GEMM_MULTI_ABD_KERNEL_HEADER)
        message(STATUS "Found GEMM Multi-ABD kernel for ctypes lib: ${GEMM_MULTI_ABD_KERNEL_HEADER}")
        add_ctypes_library(dispatcher_gemm_multi_abd_lib
            gemm_multi_abd_ctypes_lib.cpp
            KERNEL_HEADER ${GEMM_MULTI_ABD_KERNEL_HEADER}
        )
        target_compile_definitions(dispatcher_gemm_multi_abd_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE)
    else()
        message(STATUS
            "No GEMM Multi-ABD kernel found for ctypes lib - skipping dispatcher_gemm_multi_abd_lib "
            "(built at runtime by the Python bridge once a kernel header exists)")
    endif()
endif()

# --- Batched GEMM (registry-bypass, batch_count + per-batch strides ABI) ------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched_gemm_ctypes_lib.cpp)
    file(GLOB BATCHED_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*_batched.hpp")
    if(BATCHED_GEMM_KERNEL_HEADERS)
        list(SORT BATCHED_GEMM_KERNEL_HEADERS)
        list(GET BATCHED_GEMM_KERNEL_HEADERS 0 BATCHED_GEMM_KERNEL_HEADER)
        message(STATUS "Found Batched GEMM kernel for ctypes lib: ${BATCHED_GEMM_KERNEL_HEADER}")
        add_library(dispatcher_batched_gemm_lib SHARED batched_gemm_ctypes_lib.cpp)
        target_include_directories(dispatcher_batched_gemm_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
        )
        target_link_libraries(dispatcher_batched_gemm_lib PRIVATE hip::device)
        target_compile_options(dispatcher_batched_gemm_lib PRIVATE
            -include ${BATCHED_GEMM_KERNEL_HEADER}
        )
        target_compile_definitions(dispatcher_batched_gemm_lib PRIVATE CK_TILE_SINGLE_KERNEL_INCLUDE)
        set_target_properties(dispatcher_batched_gemm_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    else()
        message(STATUS "No Batched GEMM kernel found for ctypes lib - skipping dispatcher_batched_gemm_lib")
    endif()
endif()

# --- Batched-Contraction (registry-bypass, force-included kernel) ------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched_contraction_ctypes_lib.cpp)
    file(GLOB BC_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/batched_contraction_*.hpp")
    if(BC_KERNEL_HEADERS)
        list(SORT BC_KERNEL_HEADERS)
        list(GET BC_KERNEL_HEADERS 0 BC_KERNEL_HEADER)
        message(STATUS "Found batched-contraction kernel for ctypes lib: ${BC_KERNEL_HEADER}")
        add_ctypes_library(dispatcher_batched_contraction_lib
            batched_contraction_ctypes_lib.cpp
            KERNEL_HEADER ${BC_KERNEL_HEADER}
        )
        # The generated header only exports SelectedKernel/KERNEL_NAME/CONTRACTION_KEY_*
        # under this define, so the force-include path requires it.
        target_compile_definitions(dispatcher_batched_contraction_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE
        )
    else()
        message(STATUS
            "No batched-contraction kernel found for ctypes lib - skipping "
            "dispatcher_batched_contraction_lib (built at runtime by the Python "
            "bridge once a kernel header exists)")
    endif()
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

