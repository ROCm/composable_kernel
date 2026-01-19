# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(unit_sequence unit_sequence.cpp)
add_gtest_executable(unit_array unit_array.cpp)
if(result EQUAL 0)
  target_link_libraries(unit_sequence PRIVATE utility)
  target_link_libraries(unit_array PRIVATE utility)
endif()
