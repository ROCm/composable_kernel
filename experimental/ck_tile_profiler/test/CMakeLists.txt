# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(gtest)

################################################################################
# Helper Functions
################################################################################

# Helper function to create a gtest executable with common properties
function(add_ck_tile_profiler_test test_name)
    add_executable(${test_name} ${ARGN})
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_include_directories(${test_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include"
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gmock)
endfunction()


add_ck_tile_profiler_test(test_ck_tile_profiler
    test_tile_grouped_conv_bwd_data.cpp
    test_tile_grouped_conv_bwd_weight.cpp
    test_tile_grouped_conv_fwd.cpp)

include(CTest)

# Register all regression tests (integration tests with kernel compilation)
set(TILE_CONV_REGRESSION_TESTS
    test_ck_tile_profiler
)

foreach(test_target ${TILE_CONV_REGRESSION_TESTS})
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_PROFILER_REGRESSION")
endforeach()

# Helper target to build all regression tests (without running them)
add_custom_target(build-regression-ck-tile-profiler DEPENDS ${TILE_CONV_REGRESSION_TESTS})

# Target to run only regression tests (builds and runs all regression test executables)
# Use this before submitting changes to catch integration issues
add_custom_target(regression-ck-tile-profiler
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_PROFILER_REGRESSION"
    DEPENDS build-regression-ck-tile-profiler
    USES_TERMINAL
    COMMENT "Running experimental ck tile conv profiler regression tests..."
)

message(STATUS "  Regression tests: ${TILE_CONV_REGRESSION_TESTS}")
