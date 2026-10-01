# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
if(NOT GPU_TARGETS MATCHES "gfx10")
    add_executable(tile_example_batched_gemm batched_gemm.cpp)
endif()
