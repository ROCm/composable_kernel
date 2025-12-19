# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${CMAKE_CURRENT_LIST_DIR}/include/ck_tile_profiler
)
if (CK_EXPERIMENTAL_PROFILER)
    add_subdirectory(src)
endif()
if(BUILD_TESTING AND CK_EXPERIMENTAL_PROFILER_TESTS)
    add_subdirectory(test)
endif()
