# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
if(NOT GPU_TARGETS MATCHES "gfx10")
    add_executable(tile_example_gemm_multi_abd_fp16 gemm_multi_abd_fp16.cpp)
endif()
