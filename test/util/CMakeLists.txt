# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(unit_sequence unit_sequence.cpp)
if(result EQUAL 0)
  target_link_libraries(unit_sequence PRIVATE utility)
endif()

add_gtest_executable(unit_ford unit_ford.cpp)
if(result EQUAL 0)
  target_link_libraries(unit_ford PRIVATE utility)
endif()

add_gtest_executable(unit_sequence_helper unit_sequence_helper.cpp)
if(result EQUAL 0)
  target_link_libraries(unit_sequence_helper PRIVATE utility)
endif()

add_gtest_executable(unit_tensor_descriptor_functors unit_tensor_descriptor_functors.cpp)
if(result EQUAL 0)
  target_link_libraries(unit_tensor_descriptor_functors PRIVATE utility)
endif()

