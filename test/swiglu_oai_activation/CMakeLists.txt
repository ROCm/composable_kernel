# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_swiglu_oai_activation test_swiglu_oai_activation.cpp)
if(result EQUAL 0)
    target_link_libraries(test_swiglu_oai_activation PRIVATE utility)
endif()
