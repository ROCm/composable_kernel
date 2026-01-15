# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GPU verification unit tests
add_gtest_executable(test_gpu_verification test_gpu_verification.cpp)

target_link_libraries(test_gpu_verification
    PRIVATE
    utility
    device_other_operations
)
