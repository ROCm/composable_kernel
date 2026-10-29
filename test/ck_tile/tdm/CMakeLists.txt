# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx125")
   add_gtest_executable(test_tdm_basic tdm_basic.cpp)
   target_link_libraries(test_tdm_basic PRIVATE utility)

   add_gtest_executable(test_tdm_descriptor_stride tdm_descriptor_stride.cpp)
   target_link_libraries(test_tdm_descriptor_stride PRIVATE utility)

   add_gtest_executable(test_tdm_descriptor_element_size tdm_descriptor_element_size.cpp)
endif()
