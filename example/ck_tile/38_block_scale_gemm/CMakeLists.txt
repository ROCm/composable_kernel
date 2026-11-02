# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -Wno-global-constructors) # use global constructors to add kernel instances
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -enable-noalias-to-md-conversion=1")
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1")

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    set(EXE_NAME tile_example_gemm_quant)
    set(GEMM_QUANT_INSTANCE_SOURCES
        gemm_abquant_quantgrouped_fp8.cpp
        gemm_abquant_quantgrouped_fp4.cpp
        gemm_abquant_quantgrouped_bf8.cpp
        gemm_abquant_quantgrouped_preshuffleb_fp8.cpp
        gemm_abquant_quantgrouped_preshuffleb_bf8.cpp
        gemm_abquant_quantgrouped_preshuffleb_preshufflequant.cpp
        gemm_aquant_quantgrouped.cpp
        gemm_aquant_quantgrouped_preshufflequant.cpp
        gemm_bquant_quantgrouped_bf8i4.cpp
        gemm_bquant_quantgrouped_fp8i4.cpp
        gemm_bquant_quantgrouped_mx_bf16fp4.cpp
        gemm_bquant_quantgrouped_mx_bf16bf8.cpp
        gemm_bquant_quantgrouped_mx_bf16bf16.cpp
        gemm_bquant_quantgrouped_bf8.cpp
        gemm_bquant_quantgrouped_fp8.cpp
        gemm_bquant_quantgrouped_preshuffleb_bf8i4.cpp
        gemm_bquant_quantgrouped_preshuffleb_fp8i4.cpp
        gemm_bquant_quantgrouped_preshuffleb_bf8.cpp
        gemm_bquant_quantgrouped_preshuffleb_fp8.cpp
        gemm_bquant_quantgrouped_preshufflequant_bf8i4.cpp
        gemm_bquant_quantgrouped_preshufflequant_fp8i4.cpp
        gemm_bquant_quantgrouped_preshufflequant_bf8.cpp
        gemm_bquant_quantgrouped_preshufflequant_fp8.cpp
        gemm_bquant_quantgrouped_preshuffleb_preshufflequant_bf8i4.cpp
        gemm_bquant_quantgrouped_preshuffleb_preshufflequant_fp8i4.cpp
        gemm_bquant_quantgrouped_preshuffleb_preshufflequant_bf8.cpp
        gemm_bquant_quantgrouped_preshuffleb_preshufflequant_fp8.cpp
        gemm_quant_rowcol.cpp
        gemm_quant_tensor.cpp
        )
    set(GEMM_QUANT_INSTANCE_FILTER "" CACHE STRING
        "Regex selecting quantized GEMM instance source basenames (without .cpp); empty builds all")
    if(NOT GEMM_QUANT_INSTANCE_FILTER STREQUAL "")
        set(_selected_instances)
        foreach(_source IN LISTS GEMM_QUANT_INSTANCE_SOURCES)
            get_filename_component(_name "${_source}" NAME_WE)
            if(_name MATCHES "${GEMM_QUANT_INSTANCE_FILTER}")
                list(APPEND _selected_instances "${_source}")
            endif()
        endforeach()
        if(NOT _selected_instances)
            message(FATAL_ERROR "GEMM_QUANT_INSTANCE_FILTER matched no instance sources: ${GEMM_QUANT_INSTANCE_FILTER}")
        endif()
        set(GEMM_QUANT_INSTANCE_SOURCES ${_selected_instances})
    endif()
    list(LENGTH GEMM_QUANT_INSTANCE_SOURCES _instance_count)
    message(STATUS "Quantized GEMM: building ${_instance_count} instance sources")
    add_executable(${EXE_NAME} gemm_quant.cpp ${GEMM_QUANT_INSTANCE_SOURCES})
    target_compile_options(${EXE_NAME} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
