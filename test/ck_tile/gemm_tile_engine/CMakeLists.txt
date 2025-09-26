# GEMM Tile Engine Tests - Following tile_engine patterns exactly

# Set path to original tile_engine scripts  
set(TILE_ENGINE_GEMM_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm")

if(NOT EXISTS ${TILE_ENGINE_GEMM_DIR})
    message(WARNING "Tile engine directory not found: ${TILE_ENGINE_GEMM_DIR}")
    return()
endif()

# Function to create individual test target (adapted from tile_engine's create_individual_gemm_target)
function(create_individual_gemm_test_target datatype layout trait tile_config config_json)
    set(target_name "test_gemm_tile_engine_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    
    # Generate test header at build time (following tile_engine's add_custom_command pattern)
    set(test_header "${working_path}/gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    
    add_custom_command(
        OUTPUT ${test_header}
        COMMAND ${Python3_EXECUTABLE} ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
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

    # Create test executable (header NOT included as source to avoid double inclusion)
    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_simple.cpp
    )

    # Create a custom target for header generation and make test depend on it
    set(header_target "${target_name}_header")
    add_custom_target(${header_target} DEPENDS ${test_header})
    add_dependencies(${target_name} ${header_target})

    # Set compile definitions (following tile_engine's exact pattern)
    target_compile_definitions(${target_name} PRIVATE
        GEMM_SINGLE_INSTANCE_HPP="${test_header}"
    )

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TILE_ENGINE_GEMM_DIR}
        ${working_path}
    )

    # Compile options (following tile_engine's EXACT pattern with -include)
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${test_header}
    )

    message(STATUS "  Created test target: ${target_name}")
endfunction()

# Function to build test targets for datatype/layout combination (adapted from tile_engine's build_individual_gemm_targets)
function(build_gemm_test_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file (same priority as tile_engine)
    set(config_filename "smoke_test_config.json")
    set(json_blob "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_filename}")
    message(STATUS "  Using test config: ${config_filename}")

    # Check if config file exists
    if(NOT EXISTS ${json_blob})
        message(WARNING "Test config file not found: ${json_blob}")
        return()
    endif()

    # Create working directory
    file(MAKE_DIRECTORY ${working_path})

    # STEP 1: Generate kernel list using tile_engine script (following tile_engine's --list_kernels pattern)
    message(STATUS "  Listing kernel configurations for ${datatype}_${layout}...")
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

    # Read kernel count
    if(EXISTS ${working_path}/gemm_kernel_count.txt)
        file(READ ${working_path}/gemm_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(STATUS "  Found ${kernel_count} test configurations for ${datatype}_${layout}")
    else()
        message(WARNING "Kernel count file not found for ${datatype}_${layout}")
        return()
    endif()

    # STEP 2: Read kernel list and create test targets (following tile_engine's pattern)
    if(EXISTS ${working_path}/gemm_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_kernel_list.txt kernel_lines)
        set(test_count 0)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|tile_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(LENGTH parts parts_len)
            if(parts_len EQUAL 3)
                list(GET parts 0 kernel_name)
                list(GET parts 1 tile_config)
                list(GET parts 2 trait_combo)

                # Create individual test target
                create_individual_gemm_test_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
                math(EXPR test_count "${test_count} + 1")
            endif()
        endforeach()
        message(STATUS "  Created ${test_count} test targets for ${datatype}_${layout}")
    else()
        message(WARNING "Kernel list file not found for ${datatype}_${layout}")
    endif()
endfunction()

# Main execution - Create tests for fp16 with rcr layout (minimal start, following tile_engine patterns exactly)
message(STATUS "=== Starting GEMM Tile Engine Test Configuration ===")

# Check if config file exists before proceeding
set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/smoke_test_config.json")
if(EXISTS ${CONFIG_FILE})
    # Build test targets for fp16 rcr combination
    build_gemm_test_targets("fp16" "rcr")
    message(STATUS "GEMM tile engine tests configured")
else()
    message(WARNING "Config file not found: ${CONFIG_FILE}")
endif()
