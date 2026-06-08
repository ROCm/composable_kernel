# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)
set(EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})

# Currently test_ck_tile_streamk_smoke is only built on gfx9, gfx11, and gfx12 (but not gfx1250)
if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950|gfx11|gfx12" AND NOT GPU_TARGETS MATCHES "gfx1250")

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

    #TODO: support all arches
    #TODO: current c-shuffle only supports C layout as R
    add_gtest_executable(test_ck_tile_streamk_tile_partitioner test_streamk_tile_partitioner.cpp)

    # ---- Code-generate test .cpp files from types header ----
    set(STREAMK_TYPES_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_streamk_types.hpp)
    set(STREAMK_GEN_SCRIPT   ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_files.py)

    # Re-run configure automatically if the types header changes (e.g. types added/removed)
    # or if the generation script changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${STREAMK_TYPES_HEADER} ${STREAMK_GEN_SCRIPT})

    # Define the targets and their corresponding executable names
    # Note that wmma versions are suffixed with _wmma (e.g, test_ck_tile_streamk_linear_smoke_wmma)
    set(STREAMK_GEN_TARGETS extended atomic_smoke linear_smoke tree_smoke pipelines_smoke regression)
    set(STREAMK_GEN_EXEC_EXTENDED         test_ck_tile_streamk_extended)
    set(STREAMK_GEN_EXEC_ATOMIC_SMOKE     test_ck_tile_streamk_atomic_smoke)
    set(STREAMK_GEN_EXEC_LINEAR_SMOKE     test_ck_tile_streamk_linear_smoke)
    set(STREAMK_GEN_EXEC_TREE_SMOKE       test_ck_tile_streamk_tree_smoke)
    set(STREAMK_GEN_EXEC_PIPELINES_SMOKE  test_ck_tile_streamk_pipelines_smoke)
    set(STREAMK_GEN_EXEC_REGRESSION       test_ck_tile_streamk_regression)

    # Determine which variants to generate based on GPU_TARGETS.
    # add_gtest_executable handles the per-source arch filtering (removing wmma
    # sources when no gfx11/12 targets, setting HIP_ARCHITECTURES), so we only
    # need to control which .cpp files the generator emits.
    set(STREAMK_VARIANTS)
    if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950")
        list(APPEND STREAMK_VARIANTS "non-wmma")
    endif()
    # Exclude gfx1250 from WMMA support
    if(GPU_TARGETS MATCHES "gfx11|gfx12")
        list(APPEND STREAMK_VARIANTS "wmma")
    endif()

    # Collect all test targets for umbrella label
    set(CK_TILE_GEMM_STREAMK_TEST_TARGETS
        test_ck_tile_streamk_tile_partitioner)

    foreach(target IN LISTS STREAMK_GEN_TARGETS)
        string(TOUPPER ${target} TARGET_UPPER)
        set(BASE_EXEC_NAME ${STREAMK_GEN_EXEC_${TARGET_UPPER}})

        foreach(variant IN LISTS STREAMK_VARIANTS)
            if(variant STREQUAL "wmma")
                set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target}_wmma)
                set(EXEC_NAME ${BASE_EXEC_NAME}_wmma)
                set(LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}_wmma_files.txt)
                set(SOURCES_VAR ALL_SOURCES_${target}_wmma)
            else()
                set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target})
                set(EXEC_NAME ${BASE_EXEC_NAME})
                set(LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}_files.txt)
                set(SOURCES_VAR ALL_SOURCES_${target})
            endif()

            # Phase 1 (configure time): discover the list of files that will be generated
            execute_process(
                COMMAND ${Python3_EXECUTABLE} ${STREAMK_GEN_SCRIPT}
                    --types_header ${STREAMK_TYPES_HEADER}
                    --output_dir ${GEN_DIR}
                    --target ${target}
                    --variant ${variant}
                    --list_files ${LIST_FILE}
                RESULT_VARIABLE ret
                ERROR_VARIABLE list_files_stderr)
            if(ret AND NOT ret EQUAL 0)
                message(FATAL_ERROR
                    "Failed to list ${target} (${variant}) test files via Python: ${ret}\n"
                    "stderr: ${list_files_stderr}"
                )
            endif()
            file(STRINGS ${LIST_FILE} ${SOURCES_VAR})

            # Skip if no source files were generated for this variant
            if(NOT ${SOURCES_VAR})
                continue()
            endif()

            # Phase 2 (build time): generate the .cpp files when the types header changes
            add_custom_command(
                OUTPUT ${${SOURCES_VAR}}
                COMMAND ${Python3_EXECUTABLE} ${STREAMK_GEN_SCRIPT}
                    --types_header ${STREAMK_TYPES_HEADER}
                    --output_dir ${GEN_DIR}
                    --target ${target}
                    --variant ${variant}
                    --gen_files
                DEPENDS ${STREAMK_TYPES_HEADER} ${STREAMK_GEN_SCRIPT}
                COMMENT "Generating StreamK ${target} (${variant}) test sources from types header")

            # Filter out fp8/bf8 sources on gfx90a and gfx11 since those types are not natively supported
            set(FILTERED_SOURCES)
            foreach(src IN LISTS ${SOURCES_VAR})
                if(NOT src MATCHES "_(fp8|bf8)_" OR GPU_TARGETS MATCHES "gfx942|gfx950|gfx12")
                    list(APPEND FILTERED_SOURCES ${src})
                endif()
            endforeach()
            list(APPEND FILTERED_SOURCES test_gemm_streamk_util.cpp)

            add_gtest_executable(${EXEC_NAME} ${FILTERED_SOURCES})
            target_compile_options(${EXEC_NAME} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

            list(APPEND CK_TILE_GEMM_STREAMK_TEST_TARGETS ${EXEC_NAME})
        endforeach()
    endforeach()

    # Add python unit tests to validate the code gen logic in generate_test_files.py
    add_test(
        NAME test_ck_tile_streamk_generate_test_files
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_generate_test_files.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    # Label all ck_tile gemm_streamk tests with CK_TILE_GEMM_STREAMK_TESTS for selective execution
    foreach(test_target ${CK_TILE_GEMM_STREAMK_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_GEMM_STREAMK_TESTS")
    endforeach()
    # Also label the Python test
    set_tests_properties(test_ck_tile_streamk_generate_test_files PROPERTIES LABELS "CK_TILE_GEMM_STREAMK_TESTS")

    # Umbrella target to build and run all ck_tile gemm_streamk tests
    # Usage: ninja ck_tile_gemm_streamk_tests
    add_custom_target(ck_tile_gemm_streamk_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_GEMM_STREAMK_TESTS"
        DEPENDS ${CK_TILE_GEMM_STREAMK_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile gemm_streamk tests..."
    )
else()
    message(DEBUG "Skipping test_ck_tile_streamk unit tests for current target")
endif()
