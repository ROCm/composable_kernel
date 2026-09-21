# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_PADDING_K_FIRST "tile_tutorial_padding_k_first")

message(DEBUG "adding example ${EXAMPLE_PADDING_K_FIRST}")

add_executable(${EXAMPLE_PADDING_K_FIRST} EXCLUDE_FROM_ALL gemm.cpp)
target_include_directories(${EXAMPLE_PADDING_K_FIRST} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_PADDING_K_FIRST_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_PADDING_K_FIRST_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported)

target_compile_options(${EXAMPLE_PADDING_K_FIRST} PRIVATE ${EXAMPLE_PADDING_K_FIRST_COMPILE_OPTIONS})

add_dependencies(tutorials ${EXAMPLE_PADDING_K_FIRST})
