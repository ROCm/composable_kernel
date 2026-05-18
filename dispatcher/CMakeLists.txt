# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)

project(ck_tile_dispatcher VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HIP for headers (needed for validation kernels)
find_package(hip QUIET)
if(NOT hip_FOUND)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
    find_package(hip REQUIRED)
endif()

# Dispatcher library
add_library(ck_tile_dispatcher
    src/registry.cpp
    src/dispatcher.cpp
    src/fmha_registry.cpp
    src/fmha_dispatcher.cpp
)

# Enable PIC for Python bindings
set_target_properties(ck_tile_dispatcher PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ck_tile_dispatcher
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# CK Tile core headers (ck_tile/core, ck_tile/ops, etc.)
target_include_directories(ck_tile_dispatcher
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

# CK project root -- needed only for FMHA generated wrappers that include
# "example/ck_tile/01_fmha/fmha_fwd.hpp". PRIVATE to avoid exposing the
# entire project tree to downstream consumers.
target_include_directories(ck_tile_dispatcher
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# Link against HIP headers if available
if(hip_FOUND)
    target_link_libraries(ck_tile_dispatcher PUBLIC hip::host)
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ck_tile_dispatcher PRIVATE
        -Wall -Wextra -Wpedantic
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ck_tile_dispatcher PRIVATE
        /W4
    )
endif()

# Optional: Build tests
option(BUILD_DISPATCHER_TESTS "Build dispatcher unit tests" OFF)
if(BUILD_DISPATCHER_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: Build Python bindings
option(BUILD_DISPATCHER_PYTHON "Build Python bindings for dispatcher" OFF)
if(BUILD_DISPATCHER_PYTHON)
    add_subdirectory(python)
endif()

# Optional: Codegen for tile_engine integration
option(DISPATCHER_AUTO_GENERATE_WRAPPERS "Auto-generate wrappers from tile_engine" OFF)
if(DISPATCHER_AUTO_GENERATE_WRAPPERS)
    add_subdirectory(codegen)
endif()

# Optional: Build examples
option(BUILD_DISPATCHER_EXAMPLES "Build dispatcher examples" OFF)
if(BUILD_DISPATCHER_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional: Build ctypes bindings
option(BUILD_DISPATCHER_BINDINGS "Build language bindings for dispatcher" OFF)
if(BUILD_DISPATCHER_BINDINGS)
    add_subdirectory(bindings/ctypes)
endif()

# If codegen is enabled, add generated include directory
if(DISPATCHER_AUTO_GENERATE_WRAPPERS AND DISPATCHER_GENERATED_INCLUDE_DIR)
    target_include_directories(ck_tile_dispatcher
        PUBLIC
            $<BUILD_INTERFACE:${DISPATCHER_GENERATED_INCLUDE_DIR}>
    )
endif()

# Installation
install(TARGETS ck_tile_dispatcher
    EXPORT ck_tile_dispatcher_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT ck_tile_dispatcher_targets
    FILE ck_tile_dispatcher_targets.cmake
    NAMESPACE ck_tile::
    DESTINATION lib/cmake/ck_tile_dispatcher
)

