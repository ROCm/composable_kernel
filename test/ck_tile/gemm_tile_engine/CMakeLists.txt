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
    
    # Generated header path for this specific kernel configuration
    set(test_header "${working_path}/gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    
    # Generate kernel header using tile_engine's Python script
    add_custom_command(
        OUTPUT ${test_header}
        COMMAND ${Python3_EXECUTABLE} ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --gpu_target "${GEMM_TEST_GPU_TARGETS}"
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "test_gemm_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
        DEPENDS ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py ${config_json}
        COMMENT "Generating test header ${test_header}"
        VERBATIM
    )

    # Create GTest executable for this kernel configuration
    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_simple.cpp
    )

    # Ensure header is generated before compilation
    set(header_target "${target_name}_header")
    add_custom_target(${header_target} DEPENDS ${test_header})
    add_dependencies(${target_name} ${header_target})

    # Configure GPU architectures for HIP compilation
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_TEST_GPU_TARGETS})

    # Define preprocessor macros for generated header location
    target_compile_definitions(${target_name} PRIVATE
        GEMM_SINGLE_INSTANCE_HPP="${test_header}"
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
    message(STATUS "  Using test config: ${config_filename}")

    if(NOT EXISTS ${json_blob})
        message(WARNING "Test config file not found: ${json_blob}")
        return()
    endif()

    # Prepare build directory for this configuration
    file(MAKE_DIRECTORY ${working_path})

    # STEP 1: Discovery phase - list all valid kernel configurations
    message(STATUS "  Listing kernel configurations for ${datatype}_${layout}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --list_kernels
                --gpu_target "${GEMM_TEST_GPU_TARGETS}"
        WORKING_DIRECTORY ${TILE_ENGINE_GEMM_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(WARNING "Failed to list kernels for ${datatype}_${layout}: ${list_error}")
        return()
    endif()

    # Validate kernel discovery results
    if(EXISTS ${working_path}/gemm_kernel_count.txt)
        file(READ ${working_path}/gemm_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(STATUS "  Found ${kernel_count} test configurations for ${datatype}_${layout}")
    else()
        message(WARNING "Kernel count file not found for ${datatype}_${layout}")
        return()
    endif()

    # STEP 2: Generation phase - create test targets for each discovered kernel
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

# ============================================================================
# Test Configuration Matrix
# ============================================================================

# Available test configurations (minimal set for fast CI/testing)
set(TEST_CONFIGS
    "simple_test_config"
    # "medium_tiles_config"  # Uncomment for broader testing
)

# Data types for testing (core precision types)
set(TEST_DATATYPES "fp16" "bf16")
# Extended data type options:
# set(TEST_DATATYPES "fp16" "bf16" "fp32" "fp64" "int8")

# Matrix layouts for testing (row-column-row is most common)
set(TEST_LAYOUTS "rcr")
# Extended layout options:
# set(TEST_LAYOUTS "rcr" "rrr" "ccr" "crr")

# ============================================================================
# Test Target Generation Loop
# ============================================================================

foreach(datatype IN LISTS TEST_DATATYPES)
    foreach(layout IN LISTS TEST_LAYOUTS)
        foreach(config IN LISTS TEST_CONFIGS)
            set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config}.json")
            if(EXISTS ${CONFIG_FILE})
                message(STATUS "Building tests for ${datatype}_${layout}_${config}")
                build_gemm_test_targets("${datatype}" "${layout}" "${config}")
            else()
                message(WARNING "Config file not found: ${CONFIG_FILE}")
            endif()
        endforeach()
    endforeach()
endforeach()

message(STATUS "GEMM tile engine tests configured for ${TEST_DATATYPES} with ${TEST_LAYOUTS} layouts using ${TEST_CONFIGS} configurations")
