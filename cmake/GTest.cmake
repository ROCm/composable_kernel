# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

set(BUILD_GMOCK OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571
    SYSTEM
)

if(WIN32)
    set(gtest_force_shared_crt ON CACHE_INTERNAL "")
endif()

# Store the current value of BUILD_SHARED_LIBS
set(__build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(googletest)

# Restore the old value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)


if(WIN32)
    list(APPEND GTEST_CMAKE_CXX_FLAGS
            -Wno-suggest-destructor-override
            -Wno-suggest-override
            -Wno-nonportable-system-include-path
            -Wno-language-extension-token)
endif()

target_compile_options(gtest PRIVATE -Wno-undef)
target_compile_options(gtest_main PRIVATE -Wno-undef)

