# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GROUPED_GEMM_TENSORQUANT_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for Grouped TensorQuant GEMM (semicolon-separated)")
set(GROUPED_GEMM_TENSORQUANT_LAYOUT "rcr" CACHE STRING "List of layout for Grouped TensorQuant GEMM (semicolon-separated)")
set(GROUPED_GEMM_TENSORQUANT_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_GROUPED_GEMM_TENSORQUANT "Enable ccache for Grouped TensorQuant GEMM ops compilation" OFF)
set(GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES "" CACHE STRING "Max kernel instances per (dtype, layout) combo (empty = no cap)")

# Store the directory path for use in functions
set(GROUPED_GEMM_TENSORQUANT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual Grouped TensorQuant GEMM targets
function(create_individual_grouped_gemm_tensorquant_target datatype layout trait tile_config config_json)
    if(NOT GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual Grouped TensorQuant GEMM target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_grouped_gemm_tensorquant_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/grouped_gemm_tensorquant_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GROUPED_GEMM_TENSORQUANT_SOURCE_DIR}/grouped_gemm_tensorquant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "grouped_gemm_tensorquant_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GROUPED_GEMM_TENSORQUANT_SOURCE_DIR}/grouped_gemm_tensorquant_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GROUPED_GEMM_TENSORQUANT_SOURCE_DIR}/grouped_gemm_tensorquant_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GROUPED_GEMM_TENSORQUANT_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GROUPED_GEMM_TENSORQUANT_SOURCE_DIR}
        ${working_path}
    )

    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    # Add to collection targets
    add_dependencies(benchmark_grouped_gemm_tensorquant_all ${target_name})
    add_dependencies(benchmark_grouped_gemm_tensorquant_${datatype} ${target_name})
    add_dependencies(benchmark_grouped_gemm_tensorquant_${layout} ${target_name})
    add_dependencies(benchmark_grouped_gemm_tensorquant_${datatype}_${layout} ${target_name})

    # Add to trait-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_grouped_gemm_tensorquant_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_grouped_gemm_tensorquant_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_grouped_gemm_tensorquant_${scheduler}_scheduler ${target_name})
endfunction()

# Function to build individual Grouped TensorQuant GEMM targets
function(build_individual_grouped_gemm_tensorquant_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file
    if(DEFINED ENV{GROUPED_GEMM_TENSORQUANT_CONFIG_FILE} AND NOT "$ENV{GROUPED_GEMM_TENSORQUANT_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GROUPED_GEMM_TENSORQUANT_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${GROUPED_GEMM_TENSORQUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GROUPED_GEMM_TENSORQUANT_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${GROUPED_GEMM_TENSORQUANT_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    # Check if config file exists
    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Determine number of workers
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(num_workers $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        cmake_host_system_information(RESULT num_cores QUERY NUMBER_OF_LOGICAL_CORES)
        math(EXPR num_workers "${num_cores}")
        if(num_workers GREATER 8)
            set(num_workers 8)
        endif()
    endif()

    message(VERBOSE "Generating individual kernels for ${datatype} ${layout} using ${num_workers} workers...")
    message(VERBOSE "  Working path: ${working_path}")
    message(VERBOSE "  Config file: ${json_blob}")

    # Create working directory first
    file(MAKE_DIRECTORY ${working_path})

    # Buld optional args for instance builder
    set(extra_list_args "")
    if(NOT "${GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
        list(APPEND extra_list_args --tier ${TILE_ENGINE_SAMPLING_TIER})
        list(APPEND extra_list_args --manifest-path ${working_path})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_SEED}" STREQUAL "")
        list(APPEND extra_list_args --seed ${TILE_ENGINE_SAMPLING_SEED})
    endif()

    # First, just list the kernels (fast operation)
    message(VERBOSE "  Listing kernel configurations...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_tensorquant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target ${GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL}
                --list_kernels
                ${extra_list_args}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/grouped_gemm_tensorquant_kernel_count.txt)
        file(READ ${working_path}/grouped_gemm_tensorquant_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/grouped_gemm_tensorquant_kernel_list.txt)
        file(STRINGS ${working_path}/grouped_gemm_tensorquant_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|tile_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            # Create individual target
            create_individual_grouped_gemm_tensorquant_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine Grouped TensorQuant GEMM Configuration ===")
message(VERBOSE "GROUPED_GEMM_TENSORQUANT_DATATYPE: ${GROUPED_GEMM_TENSORQUANT_DATATYPE}")
message(VERBOSE "GROUPED_GEMM_TENSORQUANT_LAYOUT: ${GROUPED_GEMM_TENSORQUANT_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to only gfx942, gfx950, gfx1201
set(GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine Grouped TensorQuant GEMM build: No supported GPU targets (gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual Grouped TensorQuant targets for GPU targets: ${GROUPED_GEMM_TENSORQUANT_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4
        compile_normal=16
    )

    # Enable compiler cache if available and explicitly requested
    if(ENABLE_CCACHE_GROUPED_GEMM_TENSORQUANT)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for Grouped TensorQuant ops (use -DENABLE_CCACHE_GROUPED_GEMM_TENSORQUANT=ON to enable)")
    endif()

    # Create master collection targets
    add_custom_target(benchmark_grouped_gemm_tensorquant_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GROUPED_GEMM_TENSORQUANT_DATATYPE)
        add_custom_target(benchmark_grouped_gemm_tensorquant_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GROUPED_GEMM_TENSORQUANT_LAYOUT)
        add_custom_target(benchmark_grouped_gemm_tensorquant_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GROUPED_GEMM_TENSORQUANT_DATATYPE)
        foreach(l IN LISTS GROUPED_GEMM_TENSORQUANT_LAYOUT)
            add_custom_target(benchmark_grouped_gemm_tensorquant_${dt}_${l})
        endforeach()
    endforeach()

    # Create trait-specific collection targets (only compv3 for TensorQuant)
    add_custom_target(benchmark_grouped_gemm_tensorquant_compv3_pipeline)
    add_custom_target(benchmark_grouped_gemm_tensorquant_cshuffle_epilogue)
    add_custom_target(benchmark_grouped_gemm_tensorquant_intrawave_scheduler)

    # Divide MAX_INSTANCES budget across all active (dtype, layout) combos so that 
    # sampling fires per-combo rather than being a single cap larger than any combo's 
    # feasible set (which would make sampling a no-op for most combos).
    if(NOT "${GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES}" STREQUAL "")
        list(LENGTH GROUPED_GEMM_TENSORQUANT_DATATYPE num_datatypes)
        list(LENGTH GROUPED_GEMM_TENSORQUANT_LAYOUT num_layouts)
        math(EXPR total_combos "${num_datatypes} * ${num_layouts}")
        if(total_combos GREATER 0)
            math(EXPR GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES "${GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES} / ${total_combos}")
            message(STATUS "    grouped_gemm_tensorquant: per-combo budget = ${GROUPED_GEMM_TENSORQUANT_MAX_INSTANCES} (${total_combos} combos)")
        endif()
    endif()

    # Build targets for each datatype and layout combination
    foreach(dt IN LISTS GROUPED_GEMM_TENSORQUANT_DATATYPE)
        foreach(l IN LISTS GROUPED_GEMM_TENSORQUANT_LAYOUT)
            message(VERBOSE "Configuring Grouped TensorQuant targets for ${dt} ${l}...")
            build_individual_grouped_gemm_tensorquant_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
