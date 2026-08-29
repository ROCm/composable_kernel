# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(SUPPORTED_GPUS gfx950)

set(has_supported_gpu FALSE)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST SUPPORTED_GPUS)
        set(has_supported_gpu TRUE)
        break()
    endif()
endforeach()

if(has_supported_gpu)
    add_executable(tile_example_mx_gemm mx_gemm.cpp)
    set(EXAMPLE_MX_GEMM_COMPILE_OPTIONS -Wno-undefined-func-template)
    list(APPEND EXAMPLE_MX_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -enable-noalias-to-md-conversion=1")
    list(APPEND EXAMPLE_MX_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1")
    if(CK_USE_OCP_FP8)
        list(APPEND EXAMPLE_MX_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()
    target_compile_options(tile_example_mx_gemm PRIVATE ${EXAMPLE_MX_GEMM_COMPILE_OPTIONS})
endif()
