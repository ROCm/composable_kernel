# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_MFMA_16X16X16 "tile_tutorial_mfma_16x16x16")

message(DEBUG "adding example ${EXAMPLE_MFMA_16X16X16}")

add_executable(${EXAMPLE_MFMA_16X16X16} EXCLUDE_FROM_ALL gemm.cpp)
target_include_directories(${EXAMPLE_MFMA_16X16X16} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_MFMA_16X16X16_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_MFMA_16X16X16_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported)

target_compile_options(${EXAMPLE_MFMA_16X16X16} PRIVATE ${EXAMPLE_MFMA_16X16X16_COMPILE_OPTIONS})

add_dependencies(tutorials ${EXAMPLE_MFMA_16X16X16})
