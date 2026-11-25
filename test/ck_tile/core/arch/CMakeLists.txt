# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_subdirectory(mma)

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx9|amdgcnspirv")
    add_gtest_executable(test_arch test_arch.cpp)
    target_compile_options(test_arch PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping test_arch tests for current target")
endif()

# Ungated: the assertions are compile-time only and never instantiate the gfx1250-only
# device entry points, whose non-gfx125 branch is a hard static_assert. Still compiled as
# HIP for every SUPPORTED_GPU_TARGETS, so keep device-side instantiations out of this file.
add_gtest_executable(test_barrier test_barrier.cpp)

# Gated on the target list, not per-target: in a mixed build this still compiles for
# every GPU target, and is safe only because the device path is discarded by
# `if constexpr`. It additionally skips at runtime when the device is not gfx1250.
if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_barrier_device test_barrier_device.cpp)
    if(result EQUAL 0)
        # A broken handshake deadlocks the kernel rather than failing an assertion, so cap the
        # run instead of letting it hold a GPU for the 1500s CTest default.
        set_tests_properties(test_barrier_device PROPERTIES TIMEOUT 600)
    endif()
else()
    message(DEBUG "Skipping test_barrier_device: requires a gfx125x target")
endif()

# Compile-only, for every gfx125 variant the compiler supports, whatever GPU_TARGETS is: most
# CI configurations have no gfx125 target, so a gap in the target detection of config.hpp or
# arch.hpp would otherwise go unnoticed. A failure is a build error.
rocm_check_target_ids(CK_TILE_GFX125_TARGETS TARGETS gfx1250 gfx1250-strict gfx1251)
if(CK_TILE_GFX125_TARGETS)
    set_source_files_properties(test_compiler_target_gfx125.cpp PROPERTIES LANGUAGE HIP)
    add_library(test_ck_tile_compiler_target_gfx125 OBJECT test_compiler_target_gfx125.cpp)
    set_property(TARGET test_ck_tile_compiler_target_gfx125
        PROPERTY HIP_ARCHITECTURES ${CK_TILE_GFX125_TARGETS})
    add_dependencies(tests test_ck_tile_compiler_target_gfx125)
    add_dependencies(smoke test_ck_tile_compiler_target_gfx125)
endif()

# Ungated: split_barrier.hpp has a split-barrier path on gfx12 and a full-barrier fallback
# elsewhere, so the test builds and passes on every target. add_gtest_executable labels it
# SMOKE_TEST, which is what the gfx1250 CI stage runs ("ninja smoke"); nothing here relabels it.
add_gtest_executable(test_split_barrier test_split_barrier.cpp)
