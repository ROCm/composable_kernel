# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12|amdgcnspirv")
    add_gtest_executable(test_ck_tile_reduce2d test_reduce2d.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_threadwise test_multi_reduce2d_threadwise.cpp)
    add_gtest_executable(test_ck_tile_multi_reduce2d_multiblock test_multi_reduce2d_multiblock.cpp)
    # Exercise the gfx11 float-atomic path even on targets that default to buffer atomics.
    add_gtest_executable(test_ck_tile_multi_reduce2d_multiblock_global test_multi_reduce2d_multiblock.cpp)
    target_compile_definitions(test_ck_tile_multi_reduce2d_multiblock_global PRIVATE
        CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT=0)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_reduce2d PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_threadwise PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_multiblock PRIVATE utility)
        target_link_libraries(test_ck_tile_multi_reduce2d_multiblock_global PRIVATE utility)
    endif()

    # Collect all test targets for umbrella label
    set(CK_TILE_REDUCE_TEST_TARGETS
        test_ck_tile_reduce2d
        test_ck_tile_multi_reduce2d_threadwise
        test_ck_tile_multi_reduce2d_multiblock
        test_ck_tile_multi_reduce2d_multiblock_global
    )

    # Tail correctness must hold in both addressing and atomic backends.
    foreach(use_builtin IN ITEMS 1 0)
        foreach(use_buffer_atomics IN ITEMS 1 0)
            set(test_target test_ck_tile_multi_reduce2d_output_tails)
            if(NOT use_builtin)
                string(APPEND test_target "_legacy")
            endif()
            if(NOT use_buffer_atomics)
                string(APPEND test_target "_global")
            endif()
            add_gtest_executable(${test_target} test_multi_reduce2d_output_tails.cpp)
            target_compile_definitions(${test_target} PRIVATE
                CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN=${use_builtin})
            if(NOT use_buffer_atomics)
                target_compile_definitions(${test_target} PRIVATE
                    CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT=0)
            endif()
            list(APPEND CK_TILE_REDUCE_TEST_TARGETS ${test_target})
        endforeach()
    endforeach()

    # Label all ck_tile reduce tests with CK_TILE_REDUCE_TESTS for selective execution
    foreach(test_target ${CK_TILE_REDUCE_TEST_TARGETS})
        set_property(TEST ${test_target} APPEND PROPERTY LABELS CK_TILE_REDUCE_TESTS)
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
