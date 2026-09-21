# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/ops
    )

set(TILE_ENGINE_SAMPLING_TIER "" CACHE STRING
    "Sampling tier: 'daily' (8000 budget) or integer budget (empty = no cap)")
set(TILE_ENGINE_SAMPLING_SEED "" CACHE STRING
    "Override sampling seed (empty = daily rotation)")

add_subdirectory(ops/fmha EXCLUDE_FROM_ALL)
add_subdirectory(ops/gemm EXCLUDE_FROM_ALL)
add_subdirectory(ops/gemm_streamk EXCLUDE_FROM_ALL)
add_subdirectory(ops/pooling EXCLUDE_FROM_ALL)
add_subdirectory(ops/reduce EXCLUDE_FROM_ALL)

