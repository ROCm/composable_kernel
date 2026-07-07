# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_AQUANT_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for GEMM AQuant (semicolon-separated)")
set(GEMM_AQUANT_LAYOUT "rcr;rrr;ccr;crr" CACHE STRING "List of layout for GEMM AQuant (semicolon-separated)")
set(GEMM_AQUANT_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_GEMM_AQUANT "Enable ccache for GEMM AQuant ops compilation" OFF)

# Store the directory path for use in functions
set(GEMM_AQUANT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM AQuant targets
function(create_individual_gemm_aquant_target datatype layout trait tile_config config_json)
    if(NOT GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual GEMM AQuant target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    # Parse tile configuration: format is tile_mxtile_nxtile_k_warp_mxwarp_nxwarp_k_warp_tile_mxwarp_tile_nxwarp_tile_k
    string(REPLACE "_" ";" config_groups ${tile_config})
    list(GET config_groups 0 tile_dims)
    list(GET config_groups 1 warp_dims)
    list(GET config_groups 2 warp_tile_dims)

    string(REPLACE "x" ";" tile_parts ${tile_dims})
    list(GET tile_parts 0 tile_m)
    list(GET tile_parts 1 tile_n)
    list(GET tile_parts 2 tile_k)

    string(REPLACE "x" ";" warp_parts ${warp_dims})
    list(GET warp_parts 0 warp_m)
    list(GET warp_parts 1 warp_n)
    list(GET warp_parts 2 warp_k)

    string(REPLACE "x" ";" warp_tile_parts ${warp_tile_dims})
    list(GET warp_tile_parts 0 warp_tile_m)
    list(GET warp_tile_parts 1 warp_tile_n)
    list(GET warp_tile_parts 2 warp_tile_k)

    set(target_name "benchmark_gemm_aquant_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_aquant_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_AQUANT_SOURCE_DIR}/gemm_aquant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "gemm_aquant_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_AQUANT_SOURCE_DIR}/gemm_aquant_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_AQUANT_SOURCE_DIR}/gemm_aquant_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_AQUANT_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_AQUANT_SOURCE_DIR}
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
    add_dependencies(benchmark_gemm_aquant_all ${target_name})
    add_dependencies(benchmark_gemm_aquant_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_aquant_${layout} ${target_name})
    add_dependencies(benchmark_gemm_aquant_${datatype}_${layout} ${target_name})

    # Add to pipeline-specific and scheduler-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_gemm_aquant_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_gemm_aquant_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_gemm_aquant_${scheduler}_scheduler ${target_name})
endfunction()

# Function to build individual GEMM AQuant targets
function(build_individual_gemm_aquant_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file (priority: env var > cmake var > default)
    if(DEFINED ENV{GEMM_AQUANT_CONFIG_FILE} AND NOT "$ENV{GEMM_AQUANT_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GEMM_AQUANT_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${GEMM_AQUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_AQUANT_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${GEMM_AQUANT_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Create working directory
    file(MAKE_DIRECTORY ${working_path})

    message(VERBOSE "Generating individual AQuant kernels for ${datatype} ${layout}...")

    # Build sampling arguments
    set(extra_list_args "")
    if(NOT "${GEMM_AQUANT_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${GEMM_AQUANT_MAX_INSTANCES})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
        list(APPEND extra_list_args --tier ${TILE_ENGINE_SAMPLING_TIER})
        list(APPEND extra_list_args --manifest-path ${working_path})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_SEED}" STREQUAL "")
        list(APPEND extra_list_args --seed ${TILE_ENGINE_SAMPLING_SEED})
    endif()

    # List kernels
    message(VERBOSE "  Listing AQuant kernel configurations for ${datatype} ${layout}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_aquant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target "${GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL}"
                --list_kernels
                ${extra_list_args}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list AQuant kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/gemm_aquant_kernel_count.txt)
        file(READ ${working_path}/gemm_aquant_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} AQuant kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/gemm_aquant_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_aquant_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_aquant_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine GEMM AQuant Configuration ===")
message(VERBOSE "GEMM_AQUANT_DATATYPE: ${GEMM_AQUANT_DATATYPE}")
message(VERBOSE "GEMM_AQUANT_LAYOUT: ${GEMM_AQUANT_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to supported architectures
set(GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM AQuant build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM AQuant targets for GPU targets: ${GEMM_AQUANT_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations (directory-scoped to avoid overriding parent)
    set_property(DIRECTORY PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if requested
    if(ENABLE_CCACHE_GEMM_AQUANT)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        endif()
    endif()

    # Create master collection target
    add_custom_target(benchmark_gemm_aquant_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_AQUANT_DATATYPE)
        add_custom_target(benchmark_gemm_aquant_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GEMM_AQUANT_LAYOUT)
        add_custom_target(benchmark_gemm_aquant_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GEMM_AQUANT_DATATYPE)
        foreach(l IN LISTS GEMM_AQUANT_LAYOUT)
            add_custom_target(benchmark_gemm_aquant_${dt}_${l})
        endforeach()
    endforeach()

    # Create pipeline-specific collection targets
    set(GEMM_AQUANT_PIPELINES "mem;compv3")
    foreach(pipeline IN LISTS GEMM_AQUANT_PIPELINES)
        add_custom_target(benchmark_gemm_aquant_${pipeline}_pipeline)
    endforeach()

    # Create epilogue-specific collection targets
    set(GEMM_AQUANT_EPILOGUES "default;cshuffle")
    foreach(epilogue IN LISTS GEMM_AQUANT_EPILOGUES)
        add_custom_target(benchmark_gemm_aquant_${epilogue}_epilogue)
    endforeach()

    # Create scheduler-specific collection targets
    set(GEMM_AQUANT_SCHEDULERS "intrawave;interwave")
    foreach(scheduler IN LISTS GEMM_AQUANT_SCHEDULERS)
        add_custom_target(benchmark_gemm_aquant_${scheduler}_scheduler)
    endforeach()

    # Divide MAX_INSTANCES budget across all active (dtype, layout) combos so that
    # sampling fires per-combo rather than being a single cap.
    # CMake integer division truncates; clamp the result to at least 1 so that a
    # small total budget (e.g. 5 instances across 8 combos) does not silently
    # produce a per-combo budget of 0 and skip all kernels.
    if(NOT "${GEMM_AQUANT_MAX_INSTANCES}" STREQUAL "")
        list(LENGTH GEMM_AQUANT_DATATYPE _aq_n_dt)
        list(LENGTH GEMM_AQUANT_LAYOUT _aq_n_lay)
        math(EXPR _aq_n_combos "${_aq_n_dt} * ${_aq_n_lay}")
        if(_aq_n_combos GREATER 0)
            math(EXPR GEMM_AQUANT_MAX_INSTANCES
                "${GEMM_AQUANT_MAX_INSTANCES} / ${_aq_n_combos}")
            if(GEMM_AQUANT_MAX_INSTANCES EQUAL 0)
                set(GEMM_AQUANT_MAX_INSTANCES 1)
                message(WARNING "  gemm_aquant: per-combo budget rounded to 0; clamped to 1 (total budget too small for ${_aq_n_combos} combos)")
            else()
                message(STATUS "  gemm_aquant: per-combo budget = ${GEMM_AQUANT_MAX_INSTANCES} (${_aq_n_combos} combos)")
            endif()
        endif()
    endif()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS GEMM_AQUANT_DATATYPE)
        foreach(l IN LISTS GEMM_AQUANT_LAYOUT)
            build_individual_gemm_aquant_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
