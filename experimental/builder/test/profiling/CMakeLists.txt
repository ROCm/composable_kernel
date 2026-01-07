# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95")
  # Generate instances using python script
  find_package(Python3 COMPONENTS Interpreter Development)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_instances.py
    RESULT_VARIABLE ret
  )
  # Find cpp files and create lib for instances
  file(GLOB_RECURSE GROUPED_CONV_FWD_TILE CONFIGURE_DEPENDS "src/*.cpp")
  add_instance_library(device_grouped_conv_fwd_tile_instances ${GROUPED_CONV_FWD_TILE})

  # Add profiler executable
  set(EXAMPLE_CONV_COMPILE_OPTIONS)
  list(APPEND EXAMPLE_CONV_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0 -Wno-global-constructors -Wno-c++20-compat)
  add_executable(profile_ckb_tile_conv_fwd profile_ckb_tile_conv_fwd.cpp)
  target_compile_options(profile_ckb_tile_conv_fwd PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})
  target_compile_features(profile_ckb_tile_conv_fwd PRIVATE cxx_std_20)
  target_include_directories(profile_ckb_tile_conv_fwd PRIVATE
      "${PROJECT_SOURCE_DIR}/experimental/builder/include"
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../"
  )
  # Link with instances
  target_link_libraries(profile_ckb_tile_conv_fwd PRIVATE utility device_grouped_conv_fwd_tile_instances)

endif()
