# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_XOR_BANK_CONFLICT_FREE "tile_tutorial_xor_bank_conflict_free")

message(DEBUG "adding example ${EXAMPLE_XOR_BANK_CONFLICT_FREE}")

add_executable(${EXAMPLE_XOR_BANK_CONFLICT_FREE} EXCLUDE_FROM_ALL gemm.cpp)
target_include_directories(${EXAMPLE_XOR_BANK_CONFLICT_FREE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_XOR_BANK_CONFLICT_FREE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_XOR_BANK_CONFLICT_FREE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported)

target_compile_options(${EXAMPLE_XOR_BANK_CONFLICT_FREE} PRIVATE ${EXAMPLE_XOR_BANK_CONFLICT_FREE_COMPILE_OPTIONS})

add_dependencies(tutorials ${EXAMPLE_XOR_BANK_CONFLICT_FREE})
