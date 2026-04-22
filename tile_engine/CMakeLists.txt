# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/ops
    )

add_subdirectory(ops/gemm EXCLUDE_FROM_ALL)
add_subdirectory(ops/gemm_streamk EXCLUDE_FROM_ALL)
add_subdirectory(ops/pooling EXCLUDE_FROM_ALL)
add_subdirectory(ops/reduce EXCLUDE_FROM_ALL)

