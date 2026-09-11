# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocm_ck — constexpr schema API over CK Tile device kernels.
#
# Build from CK root:
#   cmake -B build -S . -G Ninja -DCK_ENABLE_ROCM_CK=ON
#   ninja -C build smoke-rocm-ck

# rocm_ck — header-only INTERFACE library
add_library(rocm_ck INTERFACE)
target_include_directories(rocm_ck INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# rocm_ck requires C++20, but the CK library is still migrating from c++17
# We should remove this once the library fully migrates to c++20.
target_compile_features(rocm_ck INTERFACE cxx_std_20)
target_compile_options(rocm_ck INTERFACE -Wno-c++20-compat)

# CK Tile headers — required for device code compilation
if(NOT TARGET ck_tile_headers)
    add_library(ck_tile_headers INTERFACE)
    target_include_directories(ck_tile_headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
endif()

enable_testing()
add_subdirectory(tests)
