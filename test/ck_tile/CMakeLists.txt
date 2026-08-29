# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# CK Tile Test Organization
################################################################################
# CK Tile tests can be run using several methods:
#
# 1. Global test labels (run tests across all operations):
#    - ninja smoke       - Fast tests (~30s on gfx90a)
#    - ninja regression  - Slower comprehensive tests
#    - ninja check       - All available tests
#
# 2. Operation-specific umbrella targets (run all tests for a specific operation):
#    - ninja ck_tile_gemm_tests                  - All basic GEMM tests
#    - ninja ck_tile_gemm_block_scale_tests      - All GEMM with block-scale quantization tests
#    - ninja ck_tile_gemm_streamk_tests          - All GEMM StreamK tests
#    - ninja ck_tile_grouped_gemm_quant_tests    - All grouped GEMM quantization tests
#    - ninja ck_tile_reduce_tests                - All reduce operation tests
#    - ninja ck_tile_fmha_tests                  - All FMHA (Flash Attention) tests
#
# 3. Individual test executables:
#    - ninja test_<test_name>                    - Build specific test executable
#    - ./build/bin/test_<test_name>              - Run specific test directly
#
# These umbrella targets are useful when working on specific operations to quickly
# validate all related tests without running the entire test suite.
################################################################################

add_subdirectory(image_to_column)
add_subdirectory(gemm)
add_subdirectory(gemm_persistent_async_input)
add_subdirectory(gemm_weight_preshuffle)
add_subdirectory(batched_gemm)
add_subdirectory(grouped_gemm)
add_subdirectory(grouped_gemm_preshuffle)
add_subdirectory(grouped_gemm_multi_d)
add_subdirectory(grouped_gemm_quant)
add_subdirectory(grouped_gemm_abquant)
add_subdirectory(gemm_multi_d)
add_subdirectory(gemm_multi_abd)
add_subdirectory(gemm_streamk)
add_subdirectory(data_type)
add_subdirectory(container)
add_subdirectory(elementwise)
# Not including these tests as there is a bug on gfx90a and gfx942
# resulting in "GPU core dump"
#add_subdirectory(moe_smoothquant)
add_subdirectory(permute)
add_subdirectory(moe_sorting)
add_subdirectory(slice_tile)
add_subdirectory(memory_copy)
add_subdirectory(batched_transpose)
add_subdirectory(smoothquant)
add_subdirectory(topk_softmax)
add_subdirectory(add_rmsnorm2d_rdquant)
# add_subdirectory(layernorm2d)
# add_subdirectory(rmsnorm2d)
add_subdirectory(gemm_block_scale)
add_subdirectory(flatmm)
add_subdirectory(gemm_mx)
add_subdirectory(grouped_gemm_mx)
add_subdirectory(utility)
add_subdirectory(warp_gemm)
add_subdirectory(reduce)
add_subdirectory(tdm)
add_subdirectory(core)
add_subdirectory(epilogue)
add_subdirectory(atomic_add_op)
if(BUILD_CK_TILE_FMHA_TESTS)
    add_subdirectory(fmha)
endif()
if(BUILD_CK_TILE_ENGINE_TESTS)
# TODO: The Universal GEMM tile engine test will be either removed
# or moved to the appropriate location in future work.
#   add_subdirectory(gemm_tile_engine)
    add_subdirectory(pooling_tile_engine)
endif()
add_subdirectory(pooling)
add_subdirectory(grouped_conv)
add_subdirectory(load_and_convert_tile)
add_subdirectory(multicast_load)
