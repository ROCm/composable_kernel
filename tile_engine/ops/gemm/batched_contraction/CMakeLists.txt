# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(BATCHED_CONTRACTION_DATATYPE "fp16" CACHE STRING "List of datatypes for Batched Contraction (semicolon-separated)")
set(BATCHED_CONTRACTION_LAYOUT "rcr" CACHE STRING "List of layouts for Batched Contraction (semicolon-separated, 3 chars: a+b+e)")
set(BATCHED_CONTRACTION_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(BATCHED_CONTRACTION_ELEMENTWISE_FUNCTION "PassThrough" CACHE STRING "Elementwise function for CDE epilogue")
set(BATCHED_CONTRACTION_NUM_DIM_G "1" CACHE STRING "Number of G (batch) dimensions")
set(BATCHED_CONTRACTION_NUM_DIM_M "1" CACHE STRING "Number of M dimensions")
set(BATCHED_CONTRACTION_NUM_DIM_N "1" CACHE STRING "Number of N dimensions")
set(BATCHED_CONTRACTION_NUM_DIM_K "1" CACHE STRING "Number of K dimensions")
set(BATCHED_CONTRACTION_NUM_D_TENSORS "0" CACHE STRING "Number of D (auxiliary) tensors")
set(BATCHED_CONTRACTION_MAX_INSTANCES "" CACHE STRING "Max kernel instances per (dtype, layout) combo (empty = no cap)")

option(ENABLE_CCACHE_BATCHED_CONTRACTION "Enable ccache for Batched Contraction ops compilation" OFF)

# Store the directory path for use in functions
set(BATCHED_CONTRACTION_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual Batched Contraction targets
function(create_individual_batched_contraction_target datatype layout trait tile_config config_json)
    if(NOT BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual Batched Contraction target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_batched_contraction_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/batched_contraction_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${BATCHED_CONTRACTION_SOURCE_DIR}/batched_contraction_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --num_dim_g ${BATCHED_CONTRACTION_NUM_DIM_G}
                --num_dim_m ${BATCHED_CONTRACTION_NUM_DIM_M}
                --num_dim_n ${BATCHED_CONTRACTION_NUM_DIM_N}
                --num_dim_k ${BATCHED_CONTRACTION_NUM_DIM_K}
                --num_d_tensors ${BATCHED_CONTRACTION_NUM_D_TENSORS}
                --elementwise_function ${BATCHED_CONTRACTION_ELEMENTWISE_FUNCTION}
                --config_json ${config_json}
                --gen_single
                --kernel_name "batched_contraction_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${BATCHED_CONTRACTION_SOURCE_DIR}/batched_contraction_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${BATCHED_CONTRACTION_SOURCE_DIR}/batched_contraction_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        BATCHED_CONTRACTION_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${BATCHED_CONTRACTION_SOURCE_DIR}
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
    add_dependencies(benchmark_batched_contraction_all ${target_name})
    add_dependencies(benchmark_batched_contraction_${datatype} ${target_name})
    add_dependencies(benchmark_batched_contraction_${layout} ${target_name})
    add_dependencies(benchmark_batched_contraction_${datatype}_${layout} ${target_name})

    # Add to trait-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_batched_contraction_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_batched_contraction_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_batched_contraction_${scheduler}_scheduler ${target_name})
endfunction()

# Function to build individual Batched Contraction targets
function(build_individual_batched_contraction_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file
    if(DEFINED ENV{BATCHED_CONTRACTION_CONFIG_FILE} AND NOT "$ENV{BATCHED_CONTRACTION_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{BATCHED_CONTRACTION_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${BATCHED_CONTRACTION_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${BATCHED_CONTRACTION_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${BATCHED_CONTRACTION_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    # Check if config file exists
    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Determine number of workers for parallel generation
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

    message(VERBOSE "COMMAND: ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/batched_contraction_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --num_dim_g ${BATCHED_CONTRACTION_NUM_DIM_G}
                --num_dim_m ${BATCHED_CONTRACTION_NUM_DIM_M}
                --num_dim_n ${BATCHED_CONTRACTION_NUM_DIM_N}
                --num_dim_k ${BATCHED_CONTRACTION_NUM_DIM_K}
                --num_d_tensors ${BATCHED_CONTRACTION_NUM_D_TENSORS}
                --elementwise_function ${BATCHED_CONTRACTION_ELEMENTWISE_FUNCTION}
                --config_json ${json_blob}
                --gpu_target ${BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL}
                --list_kernels ")

    set(extra_list_args "")
    if(NOT "${BATCHED_CONTRACTION_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${BATCHED_CONTRACTION_MAX_INSTANCES})
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
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/batched_contraction_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --num_dim_g ${BATCHED_CONTRACTION_NUM_DIM_G}
                --num_dim_m ${BATCHED_CONTRACTION_NUM_DIM_M}
                --num_dim_n ${BATCHED_CONTRACTION_NUM_DIM_N}
                --num_dim_k ${BATCHED_CONTRACTION_NUM_DIM_K}
                --num_d_tensors ${BATCHED_CONTRACTION_NUM_D_TENSORS}
                --elementwise_function ${BATCHED_CONTRACTION_ELEMENTWISE_FUNCTION}
                --config_json ${json_blob}
                --gpu_target ${BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL}
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
    if(EXISTS ${working_path}/batched_contraction_kernel_count.txt)
        file(READ ${working_path}/batched_contraction_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/batched_contraction_kernel_list.txt)
        file(STRINGS ${working_path}/batched_contraction_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|tile_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            # Create individual target
            create_individual_batched_contraction_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine Batched Contraction Configuration ===")
message(VERBOSE "BATCHED_CONTRACTION_DATATYPE: ${BATCHED_CONTRACTION_DATATYPE}")
message(VERBOSE "BATCHED_CONTRACTION_LAYOUT: ${BATCHED_CONTRACTION_LAYOUT}")
message(VERBOSE "BATCHED_CONTRACTION_NUM_DIM_G: ${BATCHED_CONTRACTION_NUM_DIM_G}")
message(VERBOSE "BATCHED_CONTRACTION_NUM_DIM_M: ${BATCHED_CONTRACTION_NUM_DIM_M}")
message(VERBOSE "BATCHED_CONTRACTION_NUM_DIM_N: ${BATCHED_CONTRACTION_NUM_DIM_N}")
message(VERBOSE "BATCHED_CONTRACTION_NUM_DIM_K: ${BATCHED_CONTRACTION_NUM_DIM_K}")
message(VERBOSE "BATCHED_CONTRACTION_NUM_D_TENSORS: ${BATCHED_CONTRACTION_NUM_D_TENSORS}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to only gfx90a, gfx942, gfx950
set(BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine Batched Contraction build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual Batched Contraction targets for GPU targets: ${BATCHED_CONTRACTION_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4
        compile_normal=16
    )

    # Enable compiler cache if available and explicitly requested
    if(ENABLE_CCACHE_BATCHED_CONTRACTION)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for Batched Contraction ops (use -DENABLE_CCACHE_BATCHED_CONTRACTION=ON to enable)")
    endif()

    # Create master collection targets
    add_custom_target(benchmark_batched_contraction_all)

    # Create datatype collection targets
    foreach(dt IN LISTS BATCHED_CONTRACTION_DATATYPE)
        add_custom_target(benchmark_batched_contraction_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS BATCHED_CONTRACTION_LAYOUT)
        add_custom_target(benchmark_batched_contraction_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS BATCHED_CONTRACTION_DATATYPE)
        foreach(l IN LISTS BATCHED_CONTRACTION_LAYOUT)
            add_custom_target(benchmark_batched_contraction_${dt}_${l})
        endforeach()
    endforeach()

    # Create trait-based collection targets
    set(BATCHED_CONTRACTION_PIPELINES "mem;compv3;compv4")
    set(BATCHED_CONTRACTION_EPILOGUES "default;cshuffle")
    set(BATCHED_CONTRACTION_SCHEDULERS "intrawave;interwave")

    foreach(pipeline IN LISTS BATCHED_CONTRACTION_PIPELINES)
        add_custom_target(benchmark_batched_contraction_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS BATCHED_CONTRACTION_EPILOGUES)
        add_custom_target(benchmark_batched_contraction_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS BATCHED_CONTRACTION_SCHEDULERS)
        add_custom_target(benchmark_batched_contraction_${scheduler}_scheduler)
    endforeach()

    # Divide MAX_INSTANCES budget across all active (dtype, layout) combos so that
    # sampling fires per-combo rather than being a single cap larger than any combos's
    # feasible set (which would make sampling a no-op for most combos).
    if(NOT "${BATCHED_CONTRACTION_MAX_INSTANCES}" STREQUAL "")
        list(LENGTH BATCHED_CONTRACTION_DATATYPE num_datatypes)
        list(LENGTH BATCHED_CONTRACTION_LAYOUT num_layouts)
        math(EXPR num_combos "${num_datatypes} * ${num_layouts}")
        if(num_combos GREATER 0)
            math(EXPR BATCHED_CONTRACTION_MAX_INSTANCES "${BATCHED_CONTRACTION_MAX_INSTANCES} / ${num_combos}")
            message(STATUS "    batched_contraction: per-combo budget = ${BATCHED_CONTRACTION_MAX_INSTANCES} (${num_combos} combos)")
        endif()
    endif()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS BATCHED_CONTRACTION_DATATYPE)
        foreach(l IN LISTS BATCHED_CONTRACTION_LAYOUT)
            build_individual_batched_contraction_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
