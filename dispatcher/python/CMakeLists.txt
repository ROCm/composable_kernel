# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create Python module
pybind11_add_module(_dispatcher_native bindings.cpp)

target_link_libraries(_dispatcher_native PRIVATE
    ck_tile_dispatcher
)

# Set output directory to python package location
set_target_properties(_dispatcher_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Install Python module
install(TARGETS _dispatcher_native
    LIBRARY DESTINATION python/ck_tile/dispatcher
)

install(FILES __init__.py
    DESTINATION python/ck_tile/dispatcher
)

