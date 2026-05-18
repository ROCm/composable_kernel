# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_NAIVE_GEMM "tile_tutorial_naive_gemm")

message(DEBUG "adding example ${EXAMPLE_NAIVE_GEMM}")

add_executable(${EXAMPLE_NAIVE_GEMM} EXCLUDE_FROM_ALL practice_gemm.cpp)
target_include_directories(${EXAMPLE_NAIVE_GEMM} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_NAIVE_GEMM_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_NAIVE_GEMM_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported)

target_compile_options(${EXAMPLE_NAIVE_GEMM} PRIVATE ${EXAMPLE_NAIVE_GEMM_COMPILE_OPTIONS})

add_dependencies(tutorials ${EXAMPLE_NAIVE_GEMM})
