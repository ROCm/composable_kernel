# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx1250")
  # Map GPU_TARGETS to gpu_target family for generate_instances.py
  if(GPU_TARGETS MATCHES "gfx12")
    set(GPU_TARGET_FAMILY "gfx12")
  elseif(GPU_TARGETS MATCHES "gfx11")
    set(GPU_TARGET_FAMILY "gfx11")
  else()
    set(GPU_TARGET_FAMILY "gfx9")
  endif()
  # Generate instances using python script if instance directories don't exist
  set(INSTANCES_DIR ${CMAKE_CURRENT_BINARY_DIR})
  if(NOT EXISTS ${INSTANCES_DIR}/forward OR
     NOT EXISTS ${INSTANCES_DIR}/backward_weight OR
     NOT EXISTS ${INSTANCES_DIR}/backward_data)
    find_package(Python3 COMPONENTS Interpreter Development)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_instances.py --mode=tests --instances_dir=${CMAKE_CURRENT_BINARY_DIR} --gpu_target=${GPU_TARGET_FAMILY}
      RESULT_VARIABLE ret
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error
    )

    if(NOT ret EQUAL 0)
      message(FATAL_ERROR "Failed to generate instances. Return code: ${ret}\nOutput: ${output}\nError: ${error}")
    endif()
  endif()

  # Find cpp files and create lib for instances
  file(GLOB_RECURSE GROUPED_CONV_FWD_TILE "${CMAKE_CURRENT_BINARY_DIR}/forward/*.cpp")
  add_instance_library(device_grouped_conv_fwd_tile_instances ${GROUPED_CONV_FWD_TILE})
  target_include_directories(device_grouped_conv_fwd_tile_instances PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/test/utils")
  target_compile_options(device_grouped_conv_fwd_tile_instances PRIVATE -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0 -DCK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK=1)

  file(GLOB_RECURSE GROUPED_CONV_BWD_WEIGHT_TILE "${CMAKE_CURRENT_BINARY_DIR}/backward_weight/*.cpp")
  add_instance_library(device_grouped_conv_bwd_weight_tile_instances ${GROUPED_CONV_BWD_WEIGHT_TILE})
  target_include_directories(device_grouped_conv_bwd_weight_tile_instances PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/test/utils")
  target_compile_options(device_grouped_conv_bwd_weight_tile_instances PRIVATE -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0 -DCK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK=1)

  file(GLOB_RECURSE GROUPED_CONV_BWD_DATA_TILE "${CMAKE_CURRENT_BINARY_DIR}/backward_data/*.cpp")
  add_instance_library(device_grouped_conv_bwd_data_tile_instances ${GROUPED_CONV_BWD_DATA_TILE})
  target_include_directories(device_grouped_conv_bwd_data_tile_instances PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/test/utils")
  target_compile_options(device_grouped_conv_bwd_data_tile_instances PRIVATE -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0 -DCK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK=1)
endif()
