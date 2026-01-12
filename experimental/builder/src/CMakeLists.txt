# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
  # Generate instances using python script if they are not exist
  find_package(Python3 COMPONENTS Interpreter Development)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_instances.py
    RESULT_VARIABLE ret
  )

  # Find cpp files and create lib for instances
  file(GLOB_RECURSE GROUPED_CONV_FWD_TILE "instances/*.cpp")
  add_instance_library(device_grouped_conv_fwd_tile_instances ${GROUPED_CONV_FWD_TILE})
endif()
