# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# FMHA Tile Engine -- Pure Python benchmarking via the CK dispatcher.
# No C++ per-kernel targets; all compilation is JIT via the dispatcher.

set(FMHA_TE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FMHA_TE_CONFIGS ${FMHA_TE_DIR}/configs)

include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 8)
endif()

# Use first arch from SUPPORTED_GPU_TARGETS, or fallback to gfx950
set(FMHA_BENCH_ARCH "gfx950")
if(SUPPORTED_GPU_TARGETS)
    list(GET SUPPORTED_GPU_TARGETS 0 FMHA_BENCH_ARCH)
endif()

# Main benchmark target (runs forward sweep by default)
add_custom_target(benchmark_fmha
    COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
        ${FMHA_TE_CONFIGS}/fwd.json
        --arch ${FMHA_BENCH_ARCH}
        --workers ${NPROC}
        --best
        --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_fwd_results.json
    WORKING_DIRECTORY ${FMHA_TE_DIR}
    COMMENT "FMHA tile engine benchmark (forward)"
)

if(TARGET ck_tile_dispatcher)
    add_dependencies(benchmark_fmha ck_tile_dispatcher)
endif()

# Per-variant convenience targets
foreach(variant fwd bwd splitkv appendkv pagedkv batch_prefill)
    if(EXISTS ${FMHA_TE_CONFIGS}/${variant}.json)
        add_custom_target(benchmark_fmha_${variant}
            COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
                ${FMHA_TE_CONFIGS}/${variant}.json
                --arch ${FMHA_BENCH_ARCH}
                --workers ${NPROC}
                --best
                --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_${variant}_results.json
            WORKING_DIRECTORY ${FMHA_TE_DIR}
            COMMENT "FMHA tile engine benchmark (${variant})"
        )
        if(TARGET ck_tile_dispatcher)
            add_dependencies(benchmark_fmha_${variant} ck_tile_dispatcher)
        endif()
    endif()
endforeach()

# CI target (minimal sweep for quick validation)
if(EXISTS ${FMHA_TE_CONFIGS}/fwd_ci.json)
    add_custom_target(benchmark_fmha_ci
        COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
            ${FMHA_TE_CONFIGS}/fwd_ci.json
            --arch ${FMHA_BENCH_ARCH}
            --workers 8
            --verify
        WORKING_DIRECTORY ${FMHA_TE_DIR}
        COMMENT "FMHA tile engine CI benchmark"
    )
    if(TARGET ck_tile_dispatcher)
        add_dependencies(benchmark_fmha_ci ck_tile_dispatcher)
    endif()
endif()

# All-variants target
set(FMHA_ALL_CONFIGS "")
foreach(cfg fwd bwd splitkv appendkv pagedkv batch_prefill)
    if(EXISTS ${FMHA_TE_CONFIGS}/${cfg}.json)
        list(APPEND FMHA_ALL_CONFIGS ${FMHA_TE_CONFIGS}/${cfg}.json)
    endif()
endforeach()

add_custom_target(benchmark_fmha_all
    COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
        ${FMHA_ALL_CONFIGS}
        --arch ${FMHA_BENCH_ARCH}
        --workers ${NPROC}
        --best
        --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_all_results.json
    WORKING_DIRECTORY ${FMHA_TE_DIR}
    COMMENT "FMHA tile engine benchmark (all variants)"
)

if(TARGET ck_tile_dispatcher)
    add_dependencies(benchmark_fmha_all ck_tile_dispatcher)
endif()
