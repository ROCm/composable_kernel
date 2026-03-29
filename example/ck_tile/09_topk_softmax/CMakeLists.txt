# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS)
# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

add_executable(tile_example_topk_softmax topk_softmax.cpp topk_softmax_api.cpp)
target_include_directories(tile_example_topk_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_compile_options(tile_example_topk_softmax PRIVATE ${EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS})

add_executable(tile_example_topk_softmax_decode
    topk_softmax_decode.cpp
    topk_softmax_decode_api.cpp
    topk_softmax_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../13_moe_sorting/moe_sorting_api.cpp)
target_include_directories(tile_example_topk_softmax_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../13_moe_sorting/)
target_compile_definitions(tile_example_topk_softmax_decode PRIVATE
    CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID=1
    MOE_SORTING_FMOE_2D_BUF=1)
target_compile_options(tile_example_topk_softmax_decode PRIVATE ${EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS})
