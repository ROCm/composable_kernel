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

# MX GEMM on the gfx1250 TDM V2 pipeline with a K that is not a multiple of the K tile or of the
# 32-element scale block. The TDM pipelines exist only on gfx125, so the device code is built for
# the gfx125 targets only, as for the gfx125 MX GEMM tests.
set(MX_GEMM_TDM_GPU_TARGETS ${SUPPORTED_GPU_TARGETS})
list(FILTER MX_GEMM_TDM_GPU_TARGETS INCLUDE REGEX "^gfx125")
if(MX_GEMM_TDM_GPU_TARGETS)
    set_source_files_properties(mx_gemm_tdm_k_tail.cpp PROPERTIES LANGUAGE HIP)
    add_executable(tile_example_mx_gemm_tdm_k_tail mx_gemm_tdm_k_tail.cpp)
    set_property(TARGET tile_example_mx_gemm_tdm_k_tail
                 PROPERTY HIP_ARCHITECTURES ${MX_GEMM_TDM_GPU_TARGETS})
    set(EXAMPLE_MX_GEMM_TDM_COMPILE_OPTIONS -Wno-undefined-func-template)
    if(CK_USE_OCP_FP8)
        list(APPEND EXAMPLE_MX_GEMM_TDM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()
    target_compile_options(tile_example_mx_gemm_tdm_k_tail
                           PRIVATE ${EXAMPLE_MX_GEMM_TDM_COMPILE_OPTIONS})
endif()
