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
