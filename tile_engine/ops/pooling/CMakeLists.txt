# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(POOL_DATATYPE "fp16;fp32" CACHE STRING "List of datatypes for Pool (semicolon-separated)")
set(POOL_REDUCE_OP "max;avg" CACHE STRING "List of reduce operations for Pool (semicolon-separated)")
set(POOL_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_POOL "Enable ccache for Pool ops compilation" OFF)

# Store the directory path for use in functions
set(POOL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual Pool targets
function(create_individual_pool_target datatype reduce_op trait block_config config_json)
    # Use the parent scope POOL_GPU_TARGETS_INDIVIDUAL variable
    if(NOT POOL_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual Pool target ${datatype}_${reduce_op}_${trait}_${block_config}: No supported GPU targets")
        return()
    endif()

    # Parse block configuration: format is block_mxblock_n_warp_mxwarp_n_thread_tile_mxthread_tile_n
    string(REPLACE "_" ";" config_groups ${block_config})
    list(GET config_groups 0 block_dims)       # e.g., 128x1
    list(GET config_groups 1 warp_dims)        # e.g., 1x1
    list(GET config_groups 2 thread_tile_dims) # e.g., 2x1

    # Parse block dimensions
    string(REPLACE "x" ";" block_parts ${block_dims})
    list(GET block_parts 0 block_m)
    list(GET block_parts 1 block_n)

    # Parse warp dimensions
    string(REPLACE "x" ";" warp_parts ${warp_dims})
    list(GET warp_parts 0 warp_m)
    list(GET warp_parts 1 warp_n)

    # Parse thread tile dimensions
    string(REPLACE "x" ";" thread_tile_parts ${thread_tile_dims})
    list(GET thread_tile_parts 0 thread_tile_m)
    list(GET thread_tile_parts 1 thread_tile_n)

    # Parse trait combo to get individual parts
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 output_index)
    list(GET trait_parts 1 propagate_nan)
    list(GET trait_parts 2 pool_dim)

    # Create trait string without pool_dim for filename (to match Python generator)
    set(trait_for_filename "${output_index}_${propagate_nan}")

    set(target_name "benchmark_pool${pool_dim}d_${datatype}_${reduce_op}_${trait}_${block_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${reduce_op}")

    # Generate the single instance header for this kernel (filename without pool_dim in trait)
    set(instance_header "${working_path}/pool_single_${pool_dim}d_${datatype}_${reduce_op}_${trait_for_filename}_${block_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${POOL_SOURCE_DIR}/pool_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --reduce_op ${reduce_op}
                --config_json ${config_json}
                --gen_single
                --kernel_name "pool${pool_dim}d_${datatype}_${reduce_op}_${trait_for_filename}_${block_config}"
                --block_config "${block_config}"
                --trait_combo "${trait}"
                --gpu_target "${POOL_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${POOL_SOURCE_DIR}/pool_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${POOL_SOURCE_DIR}/pool_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${POOL_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        POOL_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${POOL_SOURCE_DIR}
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
    add_dependencies(benchmark_pool_all ${target_name})
    add_dependencies(benchmark_pool_${datatype} ${target_name})
    add_dependencies(benchmark_pool_${reduce_op} ${target_name})
    add_dependencies(benchmark_pool_${datatype}_${reduce_op} ${target_name})
    add_dependencies(benchmark_pool${pool_dim}d ${target_name})
endfunction()

# Function to build individual Pool targets
function(build_individual_pool_targets datatype reduce_op)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${reduce_op}")

    # Choose config file
    if(DEFINED ENV{POOL_CONFIG_FILE} AND NOT "$ENV{POOL_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{POOL_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${POOL_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${POOL_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${POOL_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config")
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

    # Generate individual kernel files
    message(VERBOSE "Generating individual kernels for ${datatype} ${reduce_op} using ${num_workers} workers...")
    message(VERBOSE "  Working path: ${working_path}")
    message(VERBOSE "  Config file: ${json_blob}")

    # Create working directory first
    file(MAKE_DIRECTORY ${working_path})

    # List the kernels (fast operation)
    message(VERBOSE "  Listing kernel configurations...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/pool_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --reduce_op ${reduce_op}
                --config_json ${json_blob}
                --gpu_target ${POOL_GPU_TARGETS_INDIVIDUAL}
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${reduce_op}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/pool_kernel_count.txt)
        file(READ ${working_path}/pool_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/pool_kernel_list.txt)
        file(STRINGS ${working_path}/pool_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|block_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 block_config)
            list(GET parts 2 trait_combo)

            # Create individual target
            create_individual_pool_target("${datatype}" "${reduce_op}" "${trait_combo}" "${block_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine Pool Configuration ===")
message(VERBOSE "POOL_DATATYPE: ${POOL_DATATYPE}")
message(VERBOSE "POOL_REDUCE_OP: ${POOL_REDUCE_OP}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets
set(POOL_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND POOL_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT POOL_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine Pool build: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual Pool targets for GPU targets: ${POOL_GPU_TARGETS_INDIVIDUAL}")

    # Set up job pools
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4
        compile_normal=16
    )

    # Enable compiler cache if requested
    if(ENABLE_CCACHE_POOL)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for Pool ops (use -DENABLE_CCACHE_POOL=ON to enable)")
    endif()

    # Create master collection targets
    add_custom_target(benchmark_pool_all)

    # Create datatype collection targets
    foreach(dt IN LISTS POOL_DATATYPE)
        add_custom_target(benchmark_pool_${dt})
    endforeach()

    # Create reduce_op collection targets
    foreach(op IN LISTS POOL_REDUCE_OP)
        add_custom_target(benchmark_pool_${op})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS POOL_DATATYPE)
        foreach(op IN LISTS POOL_REDUCE_OP)
            add_custom_target(benchmark_pool_${dt}_${op})
        endforeach()
    endforeach()

    # Create pool dimension targets
    add_custom_target(benchmark_pool2d)
    add_custom_target(benchmark_pool3d)

    # Build individual targets for each datatype/reduce_op combination
    foreach(dt IN LISTS POOL_DATATYPE)
        foreach(op IN LISTS POOL_REDUCE_OP)
            build_individual_pool_targets(${dt} ${op})
        endforeach()
    endforeach()
endif()

