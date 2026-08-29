# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(ck_tile_unit_sequence unit_sequence.cpp)
if(result EQUAL 0)
  target_link_libraries(ck_tile_unit_sequence PRIVATE utility)
endif()
