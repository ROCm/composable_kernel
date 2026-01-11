# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
  # Find cpp files and create lib for instances
  file(GLOB_RECURSE GROUPED_CONV_FWD_TILE CONFIGURE_DEPENDS "instances/*.cpp")
  add_instance_library(device_grouped_conv_fwd_tile_instances ${GROUPED_CONV_FWD_TILE})
endif()
