# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_reduce2d test_reduce2d.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_threadwise test_multi_reduce2d_threadwise.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_multiblock test_multi_reduce2d_multiblock.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_reduce2d PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_threadwise PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_multiblock PRIVATE utility)
    endif()

    # Collect all test targets for umbrella label
    set(CK_TILE_REDUCE_TEST_TARGETS
        test_ck_tile_reduce2d
        test_ck_tile_multi_reduce2d_threadwise
        test_ck_tile_multi_reduce2d_multiblock
    )

    # Label all ck_tile reduce tests with CK_TILE_REDUCE_TESTS for selective execution
    foreach(test_target ${CK_TILE_REDUCE_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_REDUCE_TESTS")
    endforeach()

    # Umbrella target to build and run all ck_tile reduce tests
    # Usage: ninja ck_tile_reduce_tests
    add_custom_target(ck_tile_reduce_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_REDUCE_TESTS"
        DEPENDS ${CK_TILE_REDUCE_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile reduce tests..."
    )
endif()

