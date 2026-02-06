# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TARGET_NAME example_mhc)
add_executable(${TARGET_NAME} mhc.cpp)

set(TARGET_NAME example_mhc_v3)
add_executable(${TARGET_NAME} mhc_v3.cpp)

set(TARGET_NAME example_mhc_v3_single_block_test)
add_executable(${TARGET_NAME} mhc_v3_single_block_test.cpp)

set(TARGET_NAME example_mhc_v3_two_block_test)
add_executable(${TARGET_NAME} mhc_v3_two_block_test.cpp)

set(TARGET_NAME example_mhc_v3_bf16_benchmark)
add_executable(${TARGET_NAME} mhc_v3_bf16_benchmark.cpp)
