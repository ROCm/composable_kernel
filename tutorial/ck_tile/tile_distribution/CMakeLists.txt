# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# These tutorials are hard-coded for CDNA (warp_size=64) with specific tile sizes.
# Only build for gfx942 (MI300X) and gfx950 (MI350X).
if(NOT (GPU_TARGETS MATCHES "gfx942|gfx950"))
  message(VERBOSE "Skipping tile_distribution tutorials: requires gfx942 or gfx950")
  return()
endif()

foreach(i 1 2 3)
  set(TUTORIAL_NAME "tile_tutorial_tile_distribution_${i}")

  add_executable(${TUTORIAL_NAME} EXCLUDE_FROM_ALL tile_distribution_${i}.cpp)
  target_include_directories(${TUTORIAL_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_compile_options(${TUTORIAL_NAME} PRIVATE
    -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported
  )

  add_dependencies(tutorials ${TUTORIAL_NAME})
endforeach()
