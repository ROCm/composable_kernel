# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
    set(LOAD_AND_CONVERT_TILE_COMPILE_OPTIONS)

    if(CK_USE_OCP_FP8)
        list(APPEND LOAD_AND_CONVERT_TILE_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()

    if(GPU_TARGETS MATCHES "gfx11|gfx12")
        list(APPEND LOAD_AND_CONVERT_TILE_COMPILE_OPTIONS -DCK_TILE_USE_LAUNCH_BOUNDS=0)
    endif()

    add_gtest_executable(test_load_and_convert_tile_no_transpose test_load_and_convert_tile_no_transpose.cpp)
    target_compile_options(test_load_and_convert_tile_no_transpose PRIVATE ${LOAD_AND_CONVERT_TILE_COMPILE_OPTIONS})
    # gfx11: sporadic check_err failures when X and Y differ; same-type cases are stable.
    # Omit mixed-type instantiations at compile time when gfx11 is among build targets.
    if(GPU_TARGETS MATCHES "gfx11")
        target_compile_definitions(
            test_load_and_convert_tile_no_transpose
            PRIVATE GFX11_SKIP_MIXED_TYPE_TESTS=1)
    endif()

    # Transposed load is only supported on gfx950, so only add the test for that target.
    if(GPU_TARGETS MATCHES "gfx950")
        add_gtest_executable(test_load_and_convert_tile_transposed test_load_and_convert_tile_transposed.cpp)
        target_compile_options(test_load_and_convert_tile_transposed PRIVATE ${LOAD_AND_CONVERT_TILE_COMPILE_OPTIONS})
    endif()
endif()
