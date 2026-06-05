# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
include(TestUtilities)

# Helper function defined in cmake/TestUtilities.cmake to conditionally link device_conv libraries

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_grouped_convnd_bwd_data test_grouped_convnd_bwd_data.cpp)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_bwd_data PRIVATE utility device_conv_operations)

    add_executable(test_grouped_convnd_bwd_data_large_cases test_grouped_convnd_bwd_data_large_cases.cpp)
    target_compile_options(test_grouped_convnd_bwd_data_large_cases PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_bwd_data_large_cases PRIVATE gtest_main getopt::getopt utility device_conv_operations)

    add_executable(test_grouped_convnd_bwd_data_dataset_xdl test_grouped_convnd_bwd_data_dataset_xdl.cpp)
    target_compile_options(test_grouped_convnd_bwd_data_dataset_xdl PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_bwd_data_dataset_xdl PRIVATE gtest_main getopt::getopt utility device_conv_operations)

    add_executable(test_grouped_conv_bwd_data_bilinear test_grouped_conv_bwd_data_bilinear.cpp)
    target_compile_options(test_grouped_conv_bwd_data_bilinear PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_conv_bwd_data_bilinear PRIVATE gtest_main getopt::getopt utility device_conv_operations)

    add_executable(test_grouped_conv_bwd_data_scale test_grouped_conv_bwd_data_scale.cpp)
    target_compile_options(test_grouped_conv_bwd_data_scale PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_conv_bwd_data_scale PRIVATE gtest_main getopt::getopt utility device_conv_operations)
endif()

if(GPU_TARGETS MATCHES "gfx9")
  if(CK_EXPERIMENTAL_BUILDER)
    add_gtest_executable(test_grouped_convnd_bwd_data_tile test_grouped_convnd_bwd_data_tile.cpp)
    target_compile_options(test_grouped_convnd_bwd_data_tile PRIVATE -Wno-global-constructors -Wno-undef -Wno-c++20-compat)
    target_link_libraries(test_grouped_convnd_bwd_data_tile PRIVATE gtest_main getopt::getopt utility)
    if(TARGET device_grouped_conv_bwd_data_tile_instances)
        target_link_libraries(test_grouped_convnd_bwd_data_tile PRIVATE device_grouped_conv_bwd_data_tile_instances)
    endif()
  endif()
endif()

if (CK_USE_XDL OR CK_USE_WMMA) 
    add_gtest_executable(test_grouped_convnd_bwd_data_interface test_grouped_convnd_bwd_data_interface.cpp)
    if(result EQUAL 0)
        target_link_device_conv_libraries_if_exist(test_grouped_convnd_bwd_data_interface PRIVATE utility device_conv_operations)
    endif()
endif()