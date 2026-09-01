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
        ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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

# Helper function for block-scale quant ctypes targets.
#
# Usage:
#   add_quant_ctypes_target(
#     TARGET      <cmake-target-name>
#     SOURCE      <source.cpp>
#     GLOB        <glob-pattern-for-kernel-headers>
#     CACHE_VAR   <GFX_ARCH cache variable name>
#     DEFAULT_ARCH <gfx942|gfx950|FATAL_ERROR>
#     [MAIN_INCLUDE_ONLY]   # omit dispatcher/include (aquant/abquant collision fix)
#   )
#
# B1: single wrapper replacing five near-identical per-operator stanzas.
# B2: always list(SORT) the kernel glob, so the first entry is deterministic.
# D1: each plain target gets its own cache variable (CACHE_VAR), distinct from the
#     grouped targets that previously shared the same names and silently stomped the
#     plain targets' NOT DEFINED guard.
#
# MAIN_INCLUDE_ONLY: skip ${PROJECT_SOURCE_DIR}/dispatcher/include. That path pulls
#   in generated_tile_backend.hpp, which instantiates SelectedKernel::run(GemmHostArgs&)
#   and conflicts with the quant kernel's launch(QuantGemmHostArgs&). Aquant and abquant
#   set this flag; the plain GEMM and grouped quant targets do not need it.
#
# DEFAULT_ARCH: "FATAL_ERROR" makes an unresolved arch a hard configure-time error
#   (bquant). Otherwise, the value is used as the gfx fallback when neither the cache
#   variable nor CMAKE_HIP_ARCHITECTURES is set.
function(add_quant_ctypes_target)
    cmake_parse_arguments(QUANT "MAIN_INCLUDE_ONLY" "TARGET;SOURCE;GLOB;CACHE_VAR;DEFAULT_ARCH" "" ${ARGN})

    file(GLOB _headers "${QUANT_GLOB}")
    if(NOT _headers)
        message(STATUS "No kernel found for ${QUANT_TARGET} - skipping (built at runtime by Python bridge)")
        return()
    endif()

    list(SORT _headers)
    list(GET _headers 0 _kernel_header)
    message(STATUS "Found kernel for ${QUANT_TARGET}: ${_kernel_header}")

    if(QUANT_MAIN_INCLUDE_ONLY)
        # Hand-build the target without dispatcher/include to avoid the
        # SelectedKernel::run(GemmHostArgs&) vs launch(QuantGemmHostArgs&) conflict.
        add_library(${QUANT_TARGET} SHARED ${QUANT_SOURCE})
        target_include_directories(${QUANT_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${QUANT_TARGET} PRIVATE hip::device)
        target_compile_options(${QUANT_TARGET} PRIVATE -include ${_kernel_header})
        set_target_properties(${QUANT_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
    else()
        add_ctypes_library(${QUANT_TARGET} ${QUANT_SOURCE} KERNEL_HEADER ${_kernel_header})
    endif()

    # Resolve GFX arch: cache variable wins, then CMAKE_HIP_ARCHITECTURES, then DEFAULT_ARCH.
    if(NOT DEFINED ${QUANT_CACHE_VAR} OR "${${QUANT_CACHE_VAR}}" STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _resolved_arch)
            set(${QUANT_CACHE_VAR} "${_resolved_arch}" CACHE STRING
                "GFX arch for the CMake-built ${QUANT_TARGET} ctypes .so")
        elseif("${QUANT_DEFAULT_ARCH}" STREQUAL "FATAL_ERROR")
            message(FATAL_ERROR
                "${QUANT_TARGET}: GPU arch is unresolved. Set -D${QUANT_CACHE_VAR}=<gfx###> "
                "or CMAKE_HIP_ARCHITECTURES.")
        else()
            set(${QUANT_CACHE_VAR} "${QUANT_DEFAULT_ARCH}" CACHE STRING
                "GFX arch for the CMake-built ${QUANT_TARGET} ctypes .so (default: ${QUANT_DEFAULT_ARCH})")
        endif()
    endif()
    message(STATUS "${QUANT_TARGET} GFX_ARCH: ${${QUANT_CACHE_VAR}}")

    target_compile_definitions(${QUANT_TARGET} PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${${QUANT_CACHE_VAR}}"
    )
endfunction()

# =============================================================================
# GEMM ctypes Library
# =============================================================================

# Find a generated GEMM kernel header for the library.
# NOTE: this glob also matches the block-scale quant kernel headers
# (gemm_tensor_quant_*, gemm_rowcolquant_*, gemm_aquant_*, gemm_abquant_*,
# gemm_bquant_*), which define a QuantGemm SelectedKernel with an API that is
# incompatible with the regular GEMM ctypes lib. Because list(GET .. 0) takes the
# first entry and "gemm_abquant_*" sorts ahead of the regular "gemm_<variant>_*"
# names, an unfiltered glob would force-include a QuantGemm kernel into
# dispatcher_gemm_lib / gpu_helper and fail the build. Exclude the quant families.
file(GLOB GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*.hpp")
list(FILTER GEMM_KERNEL_HEADERS EXCLUDE REGEX
     "/gemm_(tensor_quant|rowcolquant|aquant|abquant|bquant)_")
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
        ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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
    ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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
        ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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
        ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_bquant_lib
    SOURCE      grouped_gemm_bquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_bquant_*.hpp"
    CACHE_VAR   CK_TILE_BQUANT_GFX_ARCH
    DEFAULT_ARCH FATAL_ERROR
)

# =============================================================================
# GroupedGemm RowColQuant ctypes Library
# =============================================================================

file(GLOB ROWCOLQUANT_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_rowcolquant_*.hpp")
if(ROWCOLQUANT_GEMM_KERNEL_HEADERS)
    list(LENGTH ROWCOLQUANT_GEMM_KERNEL_HEADERS _rowcolquant_header_count)
    if(_rowcolquant_header_count GREATER 1)
        message(WARNING
            "Found ${_rowcolquant_header_count} RowColQuant GEMM kernel headers; "
            "only the first will be compiled into dispatcher_grouped_gemm_rowcolquant_lib. "
            "Run codegen with a single config or build per-kernel .so files via the Python path.")
    endif()
    list(GET ROWCOLQUANT_GEMM_KERNEL_HEADERS 0 ROWCOLQUANT_GEMM_KERNEL_HEADER)
    message(STATUS "Found RowColQuant GEMM kernel for ctypes lib: ${ROWCOLQUANT_GEMM_KERNEL_HEADER}")

    add_ctypes_library(dispatcher_grouped_gemm_rowcolquant_lib
        grouped_gemm_rowcolquant_ctypes_lib.cpp
        KERNEL_HEADER ${ROWCOLQUANT_GEMM_KERNEL_HEADER}
    )
    if(NOT DEFINED CK_TILE_ROWCOLQUANT_GFX_ARCH OR CK_TILE_ROWCOLQUANT_GFX_ARCH STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _rowcolquant_gfx_arch)
        else()
            set(_rowcolquant_gfx_arch "gfx942")
        endif()
        set(CK_TILE_ROWCOLQUANT_GFX_ARCH "${_rowcolquant_gfx_arch}" CACHE STRING
            "GFX arch for the CMake-built RowColQuant ctypes .so (default: first CMAKE_HIP_ARCHITECTURES or gfx942)")
    endif()
    message(STATUS "RowColQuant ctypes lib GFX_ARCH: ${CK_TILE_ROWCOLQUANT_GFX_ARCH}")

    target_compile_definitions(dispatcher_grouped_gemm_rowcolquant_lib PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${CK_TILE_ROWCOLQUANT_GFX_ARCH}"
    )
else()
    message(WARNING
        "No RowColQuant GEMM kernel header found in ${CMAKE_BINARY_DIR}/generated_kernels/. "
        "dispatcher_grouped_gemm_rowcolquant_lib will NOT be built. "
        "Run unified_grouped_gemm_rowcolquant_codegen.py --output-dir "
        "${CMAKE_BINARY_DIR}/generated_kernels/ before configuring to enable this target.")
    # Do not define the target: the source requires force-included symbols (SelectedKernel,
    # ADataType, KERNEL_NAME, etc.) that only exist in the generated header. Without it
    # the build fails at compile time, not load time, breaking any all-target build.
endif()

# =============================================================================
# GroupedGemm TensorQuant ctypes Library
# =============================================================================

file(GLOB TENSORQUANT_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_tensorquant_*.hpp")
if(TENSORQUANT_GEMM_KERNEL_HEADERS)
    list(LENGTH TENSORQUANT_GEMM_KERNEL_HEADERS _tensorquant_header_count)
    if(_tensorquant_header_count GREATER 1)
        message(WARNING
            "Found ${_tensorquant_header_count} TensorQuant GEMM kernel headers; "
            "only the first will be compiled into dispatcher_grouped_gemm_tensorquant_lib. "
            "Run codegen with a single config or build per-kernel .so files via the Python path.")
    endif()
    list(GET TENSORQUANT_GEMM_KERNEL_HEADERS 0 TENSORQUANT_GEMM_KERNEL_HEADER)
    message(STATUS "Found TensorQuant GEMM kernel for ctypes lib: ${TENSORQUANT_GEMM_KERNEL_HEADER}")

    add_ctypes_library(dispatcher_grouped_gemm_tensorquant_lib
        grouped_gemm_tensorquant_ctypes_lib.cpp
        KERNEL_HEADER ${TENSORQUANT_GEMM_KERNEL_HEADER}
    )
    if(NOT DEFINED CK_TILE_TENSORQUANT_GFX_ARCH OR CK_TILE_TENSORQUANT_GFX_ARCH STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _tensorquant_gfx_arch)
        else()
            set(_tensorquant_gfx_arch "gfx942")
        endif()
        set(CK_TILE_TENSORQUANT_GFX_ARCH "${_tensorquant_gfx_arch}" CACHE STRING
            "GFX arch for the CMake-built TensorQuant ctypes .so (default: first CMAKE_HIP_ARCHITECTURES or gfx942)")
    endif()
    message(STATUS "TensorQuant ctypes lib GFX_ARCH: ${CK_TILE_TENSORQUANT_GFX_ARCH}")

    target_compile_definitions(dispatcher_grouped_gemm_tensorquant_lib PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${CK_TILE_TENSORQUANT_GFX_ARCH}"
    )
else()
    message(WARNING
        "No TensorQuant GEMM kernel header found in ${CMAKE_BINARY_DIR}/generated_kernels/. "
        "dispatcher_grouped_gemm_tensorquant_lib will NOT be built. "
        "Run unified_grouped_gemm_tensorquant_codegen.py --output-dir "
        "${CMAKE_BINARY_DIR}/generated_kernels/ before configuring to enable this target.")
    # Do not define the target: the source requires force-included symbols (SelectedKernel,
    # ADataType, KERNEL_NAME, etc.) that only exist in the generated header. Without it
    # the build fails at compile time, not load time, breaking any all-target build.
endif()

# =============================================================================
# GroupedGemm AQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_aquant_lib
    SOURCE      grouped_gemm_aquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_aquant_*.hpp"
    CACHE_VAR   CK_TILE_AQUANT_GFX_ARCH
    DEFAULT_ARCH gfx942
)

# =============================================================================
# GroupedGemm ABQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_abquant_lib
    SOURCE      grouped_gemm_abquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_abquant_*.hpp"
    CACHE_VAR   CK_TILE_ABQUANT_GFX_ARCH
    DEFAULT_ARCH gfx942
)

# =============================================================================
# Gemm TensorQuant ctypes Library
# =============================================================================
# Registry-bypass, single-kernel-per-.so bridge for the TensorQuant GEMM path
# (one scalar scale for A + one scalar scale for B). fp8/bf8, rcr layout only.
# CK_TILE_GEMM_TENSOR_QUANT_GFX_ARCH is DISTINCT from the grouped targets'
# CK_TILE_TENSOR_QUANT_GFX_ARCH to prevent the grouped `set(... CACHE)` call
# from silently pre-filling this variable before the NOT DEFINED guard below.

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_tensor_quant_lib
    SOURCE      gemm_tensor_quant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_tensor_quant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_TENSOR_QUANT_GFX_ARCH
    DEFAULT_ARCH gfx942
)

# =============================================================================
# Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
# =============================================================================
# Plain (single-problem) B-only quantized block-scale GEMM from
# 38_block_scale_gemm. Distinct from the grouped_gemm_bquant target above.
# Doubly guarded (on the generated header glob and on the source file) so the
# target is skipped cleanly if neither is present.

# =============================================================================
# Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
# =============================================================================
# Plain (single-problem) B-only quantized block-scale GEMM from
# 38_block_scale_gemm. Distinct from the grouped_gemm_bquant target above.
# CK_TILE_GEMM_BQUANT_GFX_ARCH is DISTINCT from the grouped target's
# CK_TILE_BQUANT_GFX_ARCH: the grouped block runs first and sets that variable
# via `set(... CACHE)`, which means any `if(NOT DEFINED CK_TILE_BQUANT_GFX_ARCH)`
# guard below silently never fires -- the grouped stanza's value is silently
# inherited instead. The distinct name ensures each target is independently
# configurable.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_bquant_ctypes_lib.cpp)
    add_quant_ctypes_target(
        TARGET      dispatcher_gemm_bquant_lib
        SOURCE      gemm_bquant_ctypes_lib.cpp
        GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_bquant_*.hpp"
        CACHE_VAR   CK_TILE_GEMM_BQUANT_GFX_ARCH
        DEFAULT_ARCH FATAL_ERROR
    )
    if(NOT TARGET dispatcher_gemm_bquant_lib)
        # add_quant_ctypes_target returned early (no kernel header). Provide a
        # kernel-less stub so dependents can link; it won't link fully without
        # SelectedKernel, but CMake configure/compile succeeds.
        message(STATUS "No gemm_bquant kernel found for ctypes lib - building without kernel")
        add_library(dispatcher_gemm_bquant_lib SHARED gemm_bquant_ctypes_lib.cpp)
        target_include_directories(dispatcher_gemm_bquant_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
        )
        target_link_libraries(dispatcher_gemm_bquant_lib PRIVATE hip::device)
        set_target_properties(dispatcher_gemm_bquant_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    endif()
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

# --- Batched-Contraction Multi-ABD (registry-bypass, multi-tensor + multi-dim ABI) ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched_contraction_multi_abd_ctypes_lib.cpp)
    file(GLOB CMA_KERNEL_HEADERS
        "${CMAKE_BINARY_DIR}/generated_kernels/contraction_multi_abd_*.hpp")
    if(CMA_KERNEL_HEADERS)
        list(SORT CMA_KERNEL_HEADERS)
        list(GET CMA_KERNEL_HEADERS 0 CMA_KERNEL_HEADER)
        message(STATUS "Found batched-contraction-multi-abd kernel for ctypes lib: ${CMA_KERNEL_HEADER}")

        # Resolve the GPU architecture for the CMake-driven build path.
        # The Python bridge always passes -DGFX_ARCH=... explicitly via hipcc, so
        # this only affects builds driven through CMake directly. Unlike the
        # sibling libs, there is no hard-coded fallback arch: the arch must come
        # from -DCK_TILE_CMA_GFX_ARCH or CMAKE_HIP_ARCHITECTURES, else we fail.
        if(NOT DEFINED CK_TILE_CMA_GFX_ARCH OR CK_TILE_CMA_GFX_ARCH STREQUAL "")
            if(CMAKE_HIP_ARCHITECTURES)
                list(GET CMAKE_HIP_ARCHITECTURES 0 _cma_gfx_arch)
            else()
                # Do not guess: a wrong arch compiles cleanly and then mis-tunes
                # or fails to load on the real device.
                message(FATAL_ERROR
                    "Cannot resolve a GPU architecture for "
                    "dispatcher_batched_contraction_multi_abd_lib: neither "
                    "CK_TILE_CMA_GFX_ARCH nor CMAKE_HIP_ARCHITECTURES is set. "
                    "Pass -DCK_TILE_CMA_GFX_ARCH=<gfxNNN>.")
            endif()
            set(CK_TILE_CMA_GFX_ARCH "${_cma_gfx_arch}" CACHE STRING
                "GPU architecture for dispatcher_batched_contraction_multi_abd_lib (CMake path)")
        endif()
        message(STATUS "Batched-Contraction Multi-ABD ctypes lib GFX_ARCH: ${CK_TILE_CMA_GFX_ARCH}")

        add_ctypes_library(dispatcher_batched_contraction_multi_abd_lib
            batched_contraction_multi_abd_ctypes_lib.cpp
            KERNEL_HEADER ${CMA_KERNEL_HEADER}
        )
        target_compile_definitions(dispatcher_batched_contraction_multi_abd_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE
            GFX_ARCH="${CK_TILE_CMA_GFX_ARCH}"
        )
    else()
        message(STATUS
            "No batched-contraction-multi-abd kernel found for ctypes lib - skipping "
            "dispatcher_batched_contraction_multi_abd_lib (built at runtime by the Python "
            "bridge once a kernel header exists)")
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
        ${PROJECT_SOURCE_DIR}/../include  # CK-tile core headers (ck_tile/core.hpp)
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


# =============================================================================
# Gemm RowColQuant ctypes Library
# =============================================================================
# CK_TILE_GEMM_ROWCOLQUANT_GFX_ARCH is a new name with no grouped-target alias,
# so no cache collision is possible. Included for naming consistency.

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_rowcolquant_lib
    SOURCE      gemm_rowcolquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_rowcolquant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_ROWCOLQUANT_GFX_ARCH
    DEFAULT_ARCH gfx942
)

# =============================================================================
# AQuant (A-only quantized) GEMM ctypes Library
# =============================================================================
# MAIN_INCLUDE_ONLY: omit dispatcher/include to avoid the GemmHostArgs vs
# QuantGemmHostArgs collision (see add_quant_ctypes_target docs above).
# CK_TILE_GEMM_AQUANT_GFX_ARCH is distinct from grouped CK_TILE_AQUANT_GFX_ARCH
# (D1 fix: grouped block runs first and would stomp the NOT DEFINED guard).

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_aquant_ctypes_lib.cpp)
    add_quant_ctypes_target(
        TARGET      dispatcher_gemm_aquant_lib
        SOURCE      gemm_aquant_ctypes_lib.cpp
        GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_aquant_*.hpp"
        CACHE_VAR   CK_TILE_GEMM_AQUANT_GFX_ARCH
        DEFAULT_ARCH FATAL_ERROR
        MAIN_INCLUDE_ONLY
    )
endif()

# =============================================================================
# Gemm ABQuant (A+B block-scale) ctypes Library
# =============================================================================
# MAIN_INCLUDE_ONLY: same rationale as AQuant above (QuantGemmHostArgs collision).
# CK_TILE_GEMM_ABQUANT_GFX_ARCH is distinct from grouped CK_TILE_ABQUANT_GFX_ARCH
# (D1 fix). ABQuant supports gfx942/gfx950; default gfx950 (prefill tile).

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_abquant_lib
    SOURCE      gemm_abquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_abquant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_ABQUANT_GFX_ARCH
    DEFAULT_ARCH gfx950
    MAIN_INCLUDE_ONLY
)

# =============================================================================
