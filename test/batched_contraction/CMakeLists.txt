# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if (CK_USE_XDL OR CK_USE_WMMA)
   add_gtest_executable(test_batched_contraction test_batched_contraction.cpp)
   if(result EQUAL 0)
      target_link_libraries(test_batched_contraction PRIVATE utility device_batched_gemm_bias_permute_instance)
   endif()
endif()