# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_atomic test_atomic.cpp)
add_gtest_executable(test_atomic_add_oob test_atomic_add_oob.cpp)
set(CTEST_OUTPUT_ON_FAILURE ON)
