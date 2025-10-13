# ============================================================================
# GEMM Tile Engine Unit Tests
# 
# This CMake file creates unit tests for tile_engine generated GEMM kernels.
# It follows the exact same build patterns as tile_engine for consistency
# and reliability. Each kernel configuration gets its own test executable.
# ============================================================================

# Locate tile_engine GEMM scripts directory
set(TILE_ENGINE_GEMM_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm")

if(NOT EXISTS ${TILE_ENGINE_GEMM_DIR})
    message(WARNING "Tile engine directory not found: ${TILE_ENGINE_GEMM_DIR}")
    return()
endif()

# ============================================================================
# create_individual_gemm_test_target
#
# Creates a single test executable for a specific kernel configuration.
# Mirrors tile_engine's create_individual_gemm_target function for consistency.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp32, etc.)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
#   trait        - Kernel trait combination string
#   tile_config  - Tile configuration parameters
#   config_json  - Full path to JSON configuration file
# ============================================================================
function(create_individual_gemm_test_target datatype layout config_name trait tile_config config_json)
    set(target_name "test_gemm_tile_engine_${datatype}_${layout}_${config_name}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")
    
    # Generated header path (already created during cmake configuration)
    set(test_header "${working_path}/gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    set(test_params_header "${working_path}/test_params.hpp")
    
    # Verify header exists (should have been generated during cmake configuration)
    if(NOT EXISTS ${test_header})
        message(WARNING "Generated header not found: ${test_header}")
        return()
    endif()

    # Verify test parameters header exists
    if(NOT EXISTS ${test_params_header})
        message(WARNING "Test parameters header not found: ${test_params_header}")
        return()
    endif()


    # Create GTest executable for this kernel configuration
    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_simple.cpp
    )

    # Configure GPU architectures for HIP compilation
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_TEST_GPU_TARGETS})

    # Define preprocessor macros for generated header location and test parameters
    target_compile_definitions(${target_name} PRIVATE
        GEMM_SINGLE_INSTANCE_HPP="${test_header}"
        GEMM_TEST_PARAMS_HPP="${test_params_header}"
    )

    # Include directories for headers and dependencies
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}  # Root directory for tile_engine access
        ${GTEST_INCLUDE_DIRS}
    )

    # Compiler options matching tile_engine requirements
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template  # Suppress template warnings
        -Wno-float-equal              # Allow floating point comparisons
        --offload-compress            # Enable GPU code compression
        -include ${test_header}       # Auto-include generated header
    )

    message(STATUS "  Created test target: ${target_name}")
endfunction()

# ============================================================================
# build_gemm_test_targets
#
# Builds all test targets for a specific datatype/layout/config combination.
# Uses tile_engine's two-step process: list kernels, then generate tests.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp32, etc.)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
# ============================================================================
function(build_gemm_test_targets datatype layout config_name)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")

    # Locate and validate configuration file
    set(config_filename "${config_name}.json")
    set(json_blob "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_filename}")

    if(NOT EXISTS ${json_blob})
        message(WARNING "Test config file not found: ${json_blob}")
        return()
    endif()

    # Prepare build directory for this configuration
    file(MAKE_DIRECTORY ${working_path})

    # STEP 1: Discovery phase - list all valid kernel configurations
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --list_kernels
        WORKING_DIRECTORY ${TILE_ENGINE_GEMM_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(WARNING "Failed to list kernels for ${datatype}_${layout}: ${list_error}")
        return()
    endif()

    # Check kernel count
    set(kernel_count 0)
    if(EXISTS ${working_path}/gemm_kernel_count.txt)
        file(READ ${working_path}/gemm_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
    endif()

    message(STATUS "Building tests for ${datatype}_${layout}_${config_name}")
    message(STATUS "  Found ${kernel_count} test configurations for ${datatype}_${layout}")

    # STEP 2a: Extract test parameters from config
    set(test_params_file "${working_path}/test_params.hpp")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/extract_test_params.py
                --config_file ${json_blob}
                --output_file ${test_params_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE extract_ret
        OUTPUT_VARIABLE extract_output
        ERROR_VARIABLE extract_error
    )

    if(NOT extract_ret EQUAL 0)
        message(WARNING "Failed to extract test parameters for ${datatype}_${layout}: ${extract_error}")
        return()
    endif()

    #  STEP 2b: Parallel generation phase - generate all headers at once during cmake
    if(kernel_count GREATER 0)
        message(STATUS "  Generating all headers in parallel...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                    --working_path ${working_path}
                    --datatype ${datatype}
                    --layout ${layout}
                    --config_json ${json_blob}
                    --gen_individual
            WORKING_DIRECTORY ${TILE_ENGINE_GEMM_DIR}
            RESULT_VARIABLE gen_ret
            OUTPUT_VARIABLE gen_output
            ERROR_VARIABLE gen_error
        )

        if(NOT gen_ret EQUAL 0)
            message(WARNING "Failed to generate headers for ${datatype}_${layout}: ${gen_error}")
            return()
        endif()
        message(STATUS "  All headers generated for ${datatype}_${layout}")
    endif()

    # STEP 3: Target creation phase - create test targets only if kernels were found
    if(kernel_count GREATER 0)
        if(EXISTS ${working_path}/gemm_kernel_list.txt)
            file(STRINGS ${working_path}/gemm_kernel_list.txt kernel_lines)
            set(test_count 0)
            foreach(line IN LISTS kernel_lines)
                # Parse kernel specification format: kernel_name|tile_config|trait_combo
                string(REPLACE "|" ";" parts "${line}")
                list(LENGTH parts parts_len)
                if(parts_len EQUAL 3)
                    list(GET parts 0 kernel_name)
                    list(GET parts 1 tile_config)
                    list(GET parts 2 trait_combo)

                    # Generate test target for this kernel configuration
                    create_individual_gemm_test_target("${datatype}" "${layout}" "${config_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
                    math(EXPR test_count "${test_count} + 1")
                endif()
            endforeach()
            message(STATUS "  Created ${test_count} test targets for ${datatype}_${layout}")
        else()
            message(WARNING "Kernel list file not found for ${datatype}_${layout}")
        endif()
    else()
        message(STATUS "  No test targets created (automatic validation filtered out invalid combinations)")
    endif()
endfunction()

# ============================================================================
# MAIN EXECUTION - Test Target Generation
# ============================================================================

message(STATUS "=== Starting GEMM Tile Engine Test Configuration ===")
message(STATUS "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# GPU architecture filtering - only build tests for supported architectures
set(GEMM_TEST_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_TEST_GPU_TARGETS ${target})
        message(STATUS "  Adding GPU target for tests: ${target}")
    endif()
endforeach()

# Early exit if no compatible GPU architectures are available
if(NOT GEMM_TEST_GPU_TARGETS)
    message(WARNING "Skipping GEMM Tile Engine tests: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    return()
endif()

message(STATUS "Building GEMM tile engine tests for GPU targets: ${GEMM_TEST_GPU_TARGETS}")

    # Enable parallel compilation optimizations
    # Set up job pools for better parallel compilation control
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if available and explicitly requested
    # Disabled by default due to permission issues in CI environments
    option(ENABLE_CCACHE_TESTS "Enable ccache for test compilation" OFF)
    if(ENABLE_CCACHE_TESTS)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(STATUS "Using ccache for faster test compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(STATUS "ccache disabled for tests (use -DENABLE_CCACHE_TESTS=ON to enable)")
    endif()

# ============================================================================
# Test Configuration Matrix - Clean Focused Design
# ============================================================================

# All supported data types and layouts for comprehensive testing
# Note: fp64 not included (no MFMA hardware support)
set(TEST_DATATYPES "fp16;fp8;bf16;fp32")
set(TEST_LAYOUTS "rcr;rrr;ccr;crr")

# ============================================================================
# Test Target Generation - Datatype-Specific Categories
# ============================================================================

# 1. SMALL DATATYPES: Test optimized config for small data types (fp8, fp16, bf16)
#    These data types can use larger warp tiles due to smaller memory footprint
set(SMALL_DATATYPE_CONFIG "small_datatype_config")
set(SMALL_DATATYPE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${SMALL_DATATYPE_CONFIG}.json")
set(SMALL_DATATYPES "fp8;fp16;bf16")

if(EXISTS ${SMALL_DATATYPE_CONFIG_FILE})
    message(STATUS "Processing small datatype config: ${SMALL_DATATYPE_CONFIG} (fp8, fp16, bf16)")
    foreach(datatype IN LISTS SMALL_DATATYPES)
        foreach(layout IN LISTS TEST_LAYOUTS)
            build_gemm_test_targets("${datatype}" "${layout}" "${SMALL_DATATYPE_CONFIG}")
        endforeach()
    endforeach()
else()
    message(WARNING "Small datatype config file not found: ${SMALL_DATATYPE_CONFIG_FILE}")
endif()

# 2. LARGE DATATYPES: Test optimized config for large data types (fp32)
#    These data types require smaller warp tiles due to larger memory footprint
set(LARGE_DATATYPE_CONFIG "large_datatype_config")
set(LARGE_DATATYPE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${LARGE_DATATYPE_CONFIG}.json")
set(LARGE_DATATYPES "fp32")

if(EXISTS ${LARGE_DATATYPE_CONFIG_FILE})
    message(STATUS "Processing large datatype config: ${LARGE_DATATYPE_CONFIG} (fp32)")
    foreach(datatype IN LISTS LARGE_DATATYPES)
        foreach(layout IN LISTS TEST_LAYOUTS)
            build_gemm_test_targets("${datatype}" "${layout}" "${LARGE_DATATYPE_CONFIG}")
        endforeach()
    endforeach()
else()
    message(WARNING "Large datatype config file not found: ${LARGE_DATATYPE_CONFIG_FILE}")
endif()

# 3. TRAITS COVERAGE: Test trait combinations with fixed fp16/rcr configuration
#    This focuses on kernel traits (pipelines, epilogues, schedulers, persistent)
set(TRAITS_CONFIG "traits_coverage_config")
set(TRAITS_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${TRAITS_CONFIG}.json")

if(EXISTS ${TRAITS_CONFIG_FILE})
    message(STATUS "Processing traits config: ${TRAITS_CONFIG} (fp16/rcr only)")
    build_gemm_test_targets("fp16" "rcr" "${TRAITS_CONFIG}")
else()
    message(WARNING "Traits config file not found: ${TRAITS_CONFIG_FILE}")
endif()

# 4. PADDING COVERAGE: Test padding combinations with fixed fp16/rcr configuration
#    This focuses on padding behavior (pad_m, pad_n, pad_k)
set(PADDING_CONFIG "padding_coverage_config")
set(PADDING_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${PADDING_CONFIG}.json")

if(EXISTS ${PADDING_CONFIG_FILE})
    message(STATUS "Processing padding config: ${PADDING_CONFIG} (fp16/rcr only)")
    build_gemm_test_targets("fp16" "rcr" "${PADDING_CONFIG}")
else()
    message(WARNING "Padding config file not found: ${PADDING_CONFIG_FILE}")
endif()

# 5. TILE SIZE COVERAGE: Test different tile sizes with fixed fp16/rcr configuration
#    Two levels: quick (~25 kernels) or comprehensive (~192 kernels after validation)
set(TILE_SIZE_LEVEL "quick" CACHE STRING "Tile size coverage level: quick or comprehensive")
set_property(CACHE TILE_SIZE_LEVEL PROPERTY STRINGS "quick" "comprehensive")

if(TILE_SIZE_LEVEL STREQUAL "quick")
    set(TILE_SIZE_CONFIG "tile_size_quick_config")
    set(TILE_SIZE_DESC "Quick (4×4×3 tile configs, single warp) - 48 kernels")
elseif(TILE_SIZE_LEVEL STREQUAL "comprehensive")
    set(TILE_SIZE_CONFIG "tile_size_comprehensive_config")
    set(TILE_SIZE_DESC "Comprehensive (16 tile sizes × 4 warp configs) - 1,728 kernels")
else()
    message(FATAL_ERROR "Invalid TILE_SIZE_LEVEL: ${TILE_SIZE_LEVEL}. Must be 'quick' or 'comprehensive'")
endif()

set(TILE_SIZE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${TILE_SIZE_CONFIG}.json")

if(EXISTS ${TILE_SIZE_CONFIG_FILE})
    message(STATUS "Processing tile size config: ${TILE_SIZE_LEVEL} - ${TILE_SIZE_DESC}")
    build_gemm_test_targets("fp16" "rcr" "${TILE_SIZE_CONFIG}")
else()
    message(WARNING "Tile size config file not found: ${TILE_SIZE_CONFIG_FILE}")
endif()
# ============================================================================


message(STATUS "GEMM tile engine tests configured with datatype-specific design:")
message(STATUS "  - Small datatypes (fp8, fp16, bf16) with optimized large warp tiles")
message(STATUS "  - Large datatypes (fp32) with memory-constrained smaller warp tiles")
message(STATUS "  - Traits coverage with fp16/rcr")
message(STATUS "  - Padding coverage with fp16/rcr")
message(STATUS "  - Tile size coverage: ${TILE_SIZE_LEVEL} level")
message(STATUS "    Use -DTILE_SIZE_LEVEL=comprehensive for extensive tile size testing (1,728 kernels)")
