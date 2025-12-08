# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gpu_reference_conv_fwd test_gpu_reference_conv_fwd.cpp)
target_link_libraries(test_gpu_reference_conv_fwd PRIVATE utility)
