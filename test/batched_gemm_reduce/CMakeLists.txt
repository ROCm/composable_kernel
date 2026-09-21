# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
  add_gtest_executable(test_batched_gemm_reduce_fp16 batched_gemm_reduce_fp16.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_batched_gemm_reduce_fp16 PRIVATE utility device_batched_gemm_reduce_instance)
  endif()
endif()
