# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(composable_kernel_host)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
configure_file(${CK_ROOT}/include/ck/config.h.in ${CK_ROOT}/include/ck/config.h)

find_package(ROCM)
include(ROCMInstallTargets)
include(ROCMTest)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm $ENV{ROCM_PATH})

# Debug output for hiprtc find_package issues
message(STATUS "=== hiprtc find_package debug ===")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "ROCM_PATH env: $ENV{ROCM_PATH}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    message(STATUS "Checking prefix: ${_prefix}")
    if(EXISTS "${_prefix}/lib/cmake")
        file(GLOB _cmake_pkgs "${_prefix}/lib/cmake/*")
        message(STATUS "  lib/cmake packages: ${_cmake_pkgs}")
    endif()
    if(EXISTS "${_prefix}/share")
        file(GLOB _share_dirs "${_prefix}/share/*")
        message(STATUS "  share dirs: ${_share_dirs}")
    endif()
    if(EXISTS "${_prefix}/include")
        file(GLOB _inc_dirs "${_prefix}/include/*")
        message(STATUS "  include dirs: ${_inc_dirs}")
    endif()
endforeach()

# Direct search for hiprtc files
find_path(HIPRTC_INCLUDE_TEST hiprtc/hiprtc.h PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES include NO_DEFAULT_PATH)
message(STATUS "hiprtc.h search result: ${HIPRTC_INCLUDE_TEST}")
find_library(HIPRTC_LIB_TEST hiprtc PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
message(STATUS "libhiprtc search result: ${HIPRTC_LIB_TEST}")

message(STATUS "=== end hiprtc debug ===")

find_package(hiprtc REQUIRED)

rocm_setup_version(VERSION 1.0)

list(APPEND CMAKE_MODULE_PATH ${CK_ROOT}/cmake)
include(Embed)
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
    ${CK_ROOT}/include/ck/*.hpp)

add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${CK_ROOT}/include)

add_compile_options(-std=c++20)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)
# TODO: Use object library
add_library(ck_host STATIC ${SOURCES})
target_link_libraries(ck_host PRIVATE ck_headers hiprtc::hiprtc)

set_target_properties(ck_host PROPERTIES 
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON)

# target_include_directories(ck_host PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# )

add_executable(ck-template-driver driver/main.cpp)
target_link_libraries(ck-template-driver ck_host)

rocm_install_targets(
    TARGETS ck_host ck_headers
    EXPORT ck_host_targets
    INCLUDE include
)
rocm_export_targets(
    TARGETS ck_host ck_headers
    EXPORT ck_host_targets
    NAMESPACE composable_kernel::
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

