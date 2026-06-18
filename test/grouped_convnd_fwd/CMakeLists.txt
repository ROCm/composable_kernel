# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
include(TestUtilities)

# Helper function defined in cmake/TestUtilities.cmake to conditionally link device_conv libraries

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_grouped_convnd_fwd test_grouped_convnd_fwd.cpp)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd PRIVATE utility device_conv_operations)
    add_gtest_executable(test_grouped_convnd_fwd_bilinear test_grouped_convnd_fwd_bilinear.cpp)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_bilinear PRIVATE utility device_conv_operations)

    add_gtest_executable(test_grouped_convnd_fwd_dynamic_op test_grouped_convnd_fwd_dynamic_op.cpp)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_dynamic_op PRIVATE utility device_conv_operations)

    add_gtest_executable(test_grouped_convnd_fwd_scaleadd_ab test_grouped_convnd_fwd_scaleadd_ab.cpp)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_scaleadd_ab PRIVATE utility device_conv_operations)

    add_executable(test_grouped_convnd_fwd_dataset_xdl test_grouped_convnd_fwd_dataset_xdl.cpp)
    target_compile_options(test_grouped_convnd_fwd_dataset_xdl PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_dataset_xdl PRIVATE gtest_main getopt::getopt utility device_conv_operations)

    add_executable(test_grouped_convnd_fwd_large_cases test_grouped_convnd_fwd_large_cases.cpp)
    target_compile_options(test_grouped_convnd_fwd_large_cases PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_large_cases PRIVATE gtest_main getopt::getopt utility device_conv_operations)
endif()

if(GPU_TARGETS MATCHES "gfx9|gfx1250")
  if(CK_EXPERIMENTAL_BUILDER AND NOT CK_TILE_DISPATCHER)
    add_gtest_executable(test_grouped_convnd_fwd_tile test_grouped_convnd_fwd_tile.cpp)
    target_compile_options(test_grouped_convnd_fwd_tile PRIVATE -Wno-global-constructors -Wno-undef -Wno-c++20-compat)
    target_link_libraries(test_grouped_convnd_fwd_tile PRIVATE gtest_main getopt::getopt utility)
    if(TARGET device_grouped_conv_fwd_tile_instances)
        target_link_libraries(test_grouped_convnd_fwd_tile PRIVATE device_grouped_conv_fwd_tile_instances)
    endif()
  elseif(CK_TILE_DISPATCHER)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(NOT TARGET ck_dispatcher_grouped_conv_fwd)
      add_subdirectory(
        ${PROJECT_SOURCE_DIR}/dispatcher/library
        ${CMAKE_CURRENT_BINARY_DIR}/dispatcher_library
      )
    endif()
    add_gtest_executable(test_grouped_convnd_fwd_tile test_grouped_convnd_fwd_tile.cpp)
    target_compile_options(test_grouped_convnd_fwd_tile PRIVATE
      -Wno-global-constructors -Wno-undef -Wno-c++20-compat
      -Wno-undefined-func-template -Wno-float-equal
      -Wno-ctad-maybe-unsupported -Wno-old-style-cast
      -Wno-lifetime-safety-intra-tu-suggestions)
    target_include_directories(test_grouped_convnd_fwd_tile PRIVATE
      ${PROJECT_SOURCE_DIR}/dispatcher/include
      ${PROJECT_SOURCE_DIR}/experimental/builder/include)
    target_link_libraries(test_grouped_convnd_fwd_tile PRIVATE gtest_main getopt::getopt utility)
    if(TARGET ck_dispatcher_grouped_conv_fwd)
      target_link_libraries(test_grouped_convnd_fwd_tile PRIVATE ck_dispatcher_grouped_conv_fwd)
    endif()
  endif()
endif()

add_gtest_executable(test_grouped_convnd_fwd_multi_ab_interface test_grouped_convnd_fwd_multi_ab_interface.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_convnd_fwd_multi_ab_interface PRIVATE utility)
endif()

add_gtest_executable(test_grouped_convnd_fwd_multi_d_interface_compatibility test_grouped_convnd_fwd_multi_d_interface_compatibility_xdl_wmma.cpp)
if(result EQUAL 0)
    target_link_device_conv_libraries_if_exist(test_grouped_convnd_fwd_multi_d_interface_compatibility PRIVATE utility device_conv_operations)
endif()
