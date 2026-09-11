# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configurable number of A, B, D tensors
set(GEMM_MULTI_ABD_NUM_A_TENSORS "2" CACHE STRING "Number of A tensors for gemm_multi_abd (>=1)")
set(GEMM_MULTI_ABD_NUM_B_TENSORS "2" CACHE STRING "Number of B tensors for gemm_multi_abd (>=1)")
set(GEMM_MULTI_ABD_NUM_D_TENSORS "2" CACHE STRING "Number of D tensors for gemm_multi_abd (>=1)")

set(GEMM_MULTI_ABD_DATATYPE "fp16" CACHE STRING "List of datatypes for GEMM Multi ABD (semicolon-separated)")
set(GEMM_MULTI_ABD_LAYOUT "rcrr" CACHE STRING "List of layouts for GEMM Multi ABD (semicolon-separated)")
set(GEMM_MULTI_ABD_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(GEMM_MULTI_ABD_A_ELEMENTWISE "AddScale" CACHE STRING "A elementwise function for gemm_multi_abd")
set(GEMM_MULTI_ABD_B_ELEMENTWISE "AddScale" CACHE STRING "B elementwise function for gemm_multi_abd")
set(GEMM_MULTI_ABD_CDE_ELEMENTWISE "MultiDAdd" CACHE STRING "CDE elementwise function for gemm_multi_abd")
set(GEMM_MULTI_ABD_MAX_INSTANCES "" CACHE STRING "Max kernel instances per (dtype, layout) combo (empty = no cap)")

option(ENABLE_CCACHE_GEMM_MULTI_ABD "Enable ccache for GEMM Multi ABD ops compilation" OFF)

# Store the directory path for use in functions
set(GEMM_MULTI_ABD_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM Multi ABD targets
function(create_individual_gemm_multi_abd_target datatype layout trait tile_config config_json)
    if(NOT GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual GEMM Multi ABD target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_gemm_multi_abd_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_multi_abd_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_MULTI_ABD_SOURCE_DIR}/gemm_multi_abd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --a_elementwise_function ${GEMM_MULTI_ABD_A_ELEMENTWISE}
                --b_elementwise_function ${GEMM_MULTI_ABD_B_ELEMENTWISE}
                --cde_elementwise_function ${GEMM_MULTI_ABD_CDE_ELEMENTWISE}
                --num_a_tensors ${GEMM_MULTI_ABD_NUM_A_TENSORS}
                --num_b_tensors ${GEMM_MULTI_ABD_NUM_B_TENSORS}
                --num_d_tensors ${GEMM_MULTI_ABD_NUM_D_TENSORS}
                --config_json ${config_json}
                --gen_single
                --kernel_name "gemm_multi_abd_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_MULTI_ABD_SOURCE_DIR}/gemm_multi_abd_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_MULTI_ABD_SOURCE_DIR}/gemm_multi_abd_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_MULTI_ABD_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_MULTI_ABD_SOURCE_DIR}
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
    add_dependencies(benchmark_gemm_multi_abd_all ${target_name})
    add_dependencies(benchmark_gemm_multi_abd_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_multi_abd_${layout} ${target_name})
    add_dependencies(benchmark_gemm_multi_abd_${datatype}_${layout} ${target_name})

    # Add to trait-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_gemm_multi_abd_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_gemm_multi_abd_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_gemm_multi_abd_${scheduler}_scheduler ${target_name})
endfunction()

# Function to build individual GEMM Multi ABD targets
function(build_individual_gemm_multi_abd_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Config file resolution: env var -> cmake var -> default
    if(DEFINED ENV{GEMM_MULTI_ABD_CONFIG_FILE} AND NOT "$ENV{GEMM_MULTI_ABD_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GEMM_MULTI_ABD_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${GEMM_MULTI_ABD_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_MULTI_ABD_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${GEMM_MULTI_ABD_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

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

    # Create working directory
    file(MAKE_DIRECTORY ${working_path})

    message(VERBOSE "Generating individual kernels for ${datatype} ${layout} using ${num_workers} workers...")
    message(VERBOSE "  NumA=${GEMM_MULTI_ABD_NUM_A_TENSORS}, NumB=${GEMM_MULTI_ABD_NUM_B_TENSORS}, NumD=${GEMM_MULTI_ABD_NUM_D_TENSORS}")
    message(VERBOSE "  Working path: ${working_path}")
    message(VERBOSE "  Config file: ${json_blob}")
    message(VERBOSE "  Python executable: ${Python3_EXECUTABLE}")
    message(VERBOSE "  Script path: ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_abd_instance_builder.py")

    set(extra_list_args "")
    if(NOT "${GEMM_MULTI_ABD_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${GEMM_MULTI_ABD_MAX_INSTANCES})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
        list(APPEND extra_list_args --tier ${TILE_ENGINE_SAMPLING_TIER})
        list(APPEND extra_list_args --manifest-path ${working_path})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_SEED}" STREQUAL "")
        list(APPEND extra_list_args --seed ${TILE_ENGINE_SAMPLING_SEED})
    endif()

    # List kernels (fast configure-time operation)
    message(VERBOSE "  Listing kernel configurations...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_abd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --a_elementwise_function ${GEMM_MULTI_ABD_A_ELEMENTWISE}
                --b_elementwise_function ${GEMM_MULTI_ABD_B_ELEMENTWISE}
                --cde_elementwise_function ${GEMM_MULTI_ABD_CDE_ELEMENTWISE}
                --num_a_tensors ${GEMM_MULTI_ABD_NUM_A_TENSORS}
                --num_b_tensors ${GEMM_MULTI_ABD_NUM_B_TENSORS}
                --num_d_tensors ${GEMM_MULTI_ABD_NUM_D_TENSORS}
                --config_json ${json_blob}
                --gpu_target ${GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}
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
    if(EXISTS ${working_path}/gemm_multi_abd_kernel_count.txt)
        file(READ ${working_path}/gemm_multi_abd_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/gemm_multi_abd_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_multi_abd_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|tile_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_multi_abd_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine GEMM Multi ABD Configuration ===")
message(VERBOSE "GEMM_MULTI_ABD_DATATYPE: ${GEMM_MULTI_ABD_DATATYPE}")
message(VERBOSE "GEMM_MULTI_ABD_LAYOUT: ${GEMM_MULTI_ABD_LAYOUT}")
message(VERBOSE "GEMM_MULTI_ABD_NUM_A_TENSORS: ${GEMM_MULTI_ABD_NUM_A_TENSORS}")
message(VERBOSE "GEMM_MULTI_ABD_NUM_B_TENSORS: ${GEMM_MULTI_ABD_NUM_B_TENSORS}")
message(VERBOSE "GEMM_MULTI_ABD_NUM_D_TENSORS: ${GEMM_MULTI_ABD_NUM_D_TENSORS}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to only gfx90a, gfx942, gfx950, gfx1201
set(GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM Multi ABD build: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM Multi ABD targets for GPU targets: ${GEMM_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4
        compile_normal=16
    )

    # Enable compiler cache if available and explicitly requested
    if(ENABLE_CCACHE_GEMM_MULTI_ABD)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for GEMM Multi ABD ops (use -DENABLE_CCACHE_GEMM_MULTI_ABD=ON to enable)")
    endif()

    # Create master collection target
    add_custom_target(benchmark_gemm_multi_abd_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_MULTI_ABD_DATATYPE)
        add_custom_target(benchmark_gemm_multi_abd_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GEMM_MULTI_ABD_LAYOUT)
        add_custom_target(benchmark_gemm_multi_abd_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GEMM_MULTI_ABD_DATATYPE)
        foreach(l IN LISTS GEMM_MULTI_ABD_LAYOUT)
            add_custom_target(benchmark_gemm_multi_abd_${dt}_${l})
        endforeach()
    endforeach()

    # Create trait-based collection targets
    set(GEMM_MULTI_ABD_PIPELINES "mem;compv3;compv4")
    set(GEMM_MULTI_ABD_EPILOGUES "default;cshuffle")
    set(GEMM_MULTI_ABD_SCHEDULERS "intrawave;interwave")

    foreach(pipeline IN LISTS GEMM_MULTI_ABD_PIPELINES)
        add_custom_target(benchmark_gemm_multi_abd_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS GEMM_MULTI_ABD_EPILOGUES)
        add_custom_target(benchmark_gemm_multi_abd_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS GEMM_MULTI_ABD_SCHEDULERS)
        add_custom_target(benchmark_gemm_multi_abd_${scheduler}_scheduler)
    endforeach()

    # Divide MAX_INSTANCES budget across all active (dtype, layout) combos so that
    # sampling fires per-combo rather than being a single cap larger than any combo's
    # feasible set (which would make sampling a no-op for most combos).
    if(NOT "${GEMM_MULTI_ABD_MAX_INSTANCES}" STREQUAL "")
        list(LENGTH GEMM_MULTI_ABD_DATATYPE num_datatypes)
        list(LENGTH GEMM_MULTI_ABD_LAYOUT num_layouts)
        math(EXPR num_combos "${num_datatypes} * ${num_layouts}")
        if(num_combos GREATER 0)
            math(EXPR GEMM_MULTI_ABD_MAX_INSTANCES "${GEMM_MULTI_ABD_MAX_INSTANCES} / ${num_combos}")
            message(STATUS "    gemm_multi_abd: per-combo budget = ${GEMM_MULTI_ABD_MAX_INSTANCES} (${num_combos} combos)")
        endif()
    endif()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS GEMM_MULTI_ABD_DATATYPE)
        foreach(l IN LISTS GEMM_MULTI_ABD_LAYOUT)
            build_individual_gemm_multi_abd_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
