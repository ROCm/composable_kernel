# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${CMAKE_CURRENT_LIST_DIR}/include/ck_tile_profiler
)
add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
