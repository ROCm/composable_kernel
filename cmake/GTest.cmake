# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

set(BUILD_GMOCK OFF CACHE INTERNAL "")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        b85864c64758dec007208e56af933fc3f52044ee
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

set(GTEST_CMAKE_CXX_FLAGS
         -Wno-undef
         -Wno-reserved-identifier
         -Wno-global-constructors
         -Wno-missing-noreturn
         -Wno-disabled-macro-expansion
         -Wno-used-but-marked-unused
         -Wno-switch-enum
         -Wno-zero-as-null-pointer-constant
         -Wno-unused-member-function
         -Wno-comma
         -Wno-old-style-cast
         -Wno-deprecated
         -Wno-unsafe-buffer-usage)

if(WIN32)
    list(APPEND GTEST_CMAKE_CXX_FLAGS
            -Wno-suggest-destructor-override
            -Wno-suggest-override
            -Wno-nonportable-system-include-path
            -Wno-language-extension-token)
endif()

target_compile_options(gtest PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
if(WIN32)
   target_compile_definitions(gtest PUBLIC GTEST_HAS_SEH=0)
   target_compile_definitions(gtest_main PUBLIC GTEST_HAS_SEH=0)
endif()

include(GoogleTest)
unset(GTEST_CMAKE_CXX_FLAGS)
