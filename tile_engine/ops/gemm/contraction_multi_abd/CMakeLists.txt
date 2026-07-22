# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CONTRACTION_MULTI_ABD_DATATYPE "fp16" CACHE STRING "List of datatypes for Contraction Multi ABD (semicolon-separated)")
set(CONTRACTION_MULTI_ABD_LAYOUT "rcr" CACHE STRING "List of layouts for Contraction Multi ABD (semicolon-separated, 3-char: A,B,E)")
set(CONTRACTION_MULTI_ABD_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(CONTRACTION_MULTI_ABD_A_ELEMENTWISE "PassThrough" CACHE STRING "A elementwise function")
set(CONTRACTION_MULTI_ABD_B_ELEMENTWISE "PassThrough" CACHE STRING "B elementwise function")
set(CONTRACTION_MULTI_ABD_CDE_ELEMENTWISE "AddDs" CACHE STRING "CDE elementwise function")
set(CONTRACTION_MULTI_ABD_NUM_A_TENSOR "2" CACHE STRING "Number of A tensors")
set(CONTRACTION_MULTI_ABD_NUM_B_TENSOR "1" CACHE STRING "Number of B tensors")
set(CONTRACTION_MULTI_ABD_NUM_D_TENSOR "1" CACHE STRING "Number of D tensors")
set(CONTRACTION_MULTI_ABD_NUM_DIM_G "1" CACHE STRING "Number of G dimensions")
set(CONTRACTION_MULTI_ABD_NUM_DIM_M "1" CACHE STRING "Number of M dimensions")
set(CONTRACTION_MULTI_ABD_NUM_DIM_N "1" CACHE STRING "Number of N dimensions")
set(CONTRACTION_MULTI_ABD_NUM_DIM_K "1" CACHE STRING "Number of K dimensions")

option(ENABLE_CCACHE_CONTRACTION_MULTI_ABD "Enable ccache for Contraction Multi ABD compilation" OFF)

set(CONTRACTION_MULTI_ABD_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(create_individual_contraction_multi_abd_target datatype layout trait tile_config config_json)
    if(NOT CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual Contraction Multi ABD target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

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

    set(target_name "benchmark_contraction_multi_abd_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    set(instance_header "${working_path}/contraction_multi_abd_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${CONTRACTION_MULTI_ABD_SOURCE_DIR}/contraction_multi_abd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --a_elementwise ${CONTRACTION_MULTI_ABD_A_ELEMENTWISE}
                --b_elementwise ${CONTRACTION_MULTI_ABD_B_ELEMENTWISE}
                --cde_elementwise ${CONTRACTION_MULTI_ABD_CDE_ELEMENTWISE}
                --num_a_tensor ${CONTRACTION_MULTI_ABD_NUM_A_TENSOR}
                --num_b_tensor ${CONTRACTION_MULTI_ABD_NUM_B_TENSOR}
                --num_d_tensor ${CONTRACTION_MULTI_ABD_NUM_D_TENSOR}
                --num_dim_g ${CONTRACTION_MULTI_ABD_NUM_DIM_G}
                --num_dim_m ${CONTRACTION_MULTI_ABD_NUM_DIM_M}
                --num_dim_n ${CONTRACTION_MULTI_ABD_NUM_DIM_N}
                --num_dim_k ${CONTRACTION_MULTI_ABD_NUM_DIM_K}
                --config_json ${config_json}
                --gen_single
                --kernel_name "contraction_multi_abd_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${CONTRACTION_MULTI_ABD_SOURCE_DIR}/contraction_multi_abd_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${CONTRACTION_MULTI_ABD_SOURCE_DIR}/contraction_multi_abd_benchmark_single.cpp
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL})

    target_compile_definitions(${target_name} PRIVATE
        CONTRACTION_MULTI_ABD_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    target_include_directories(${target_name} PRIVATE
        ${CONTRACTION_MULTI_ABD_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    add_dependencies(benchmark_contraction_multi_abd_all ${target_name})
    add_dependencies(benchmark_contraction_multi_abd_${datatype} ${target_name})
    add_dependencies(benchmark_contraction_multi_abd_${layout} ${target_name})
    add_dependencies(benchmark_contraction_multi_abd_${datatype}_${layout} ${target_name})

    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_contraction_multi_abd_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_contraction_multi_abd_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_contraction_multi_abd_${scheduler}_scheduler ${target_name})
endfunction()

function(build_individual_contraction_multi_abd_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if(DEFINED ENV{CONTRACTION_MULTI_ABD_CONFIG_FILE} AND NOT "$ENV{CONTRACTION_MULTI_ABD_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{CONTRACTION_MULTI_ABD_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${CONTRACTION_MULTI_ABD_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${CONTRACTION_MULTI_ABD_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${CONTRACTION_MULTI_ABD_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(num_workers $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        cmake_host_system_information(RESULT num_cores QUERY NUMBER_OF_LOGICAL_CORES)
        math(EXPR num_workers "${num_cores}")
        if(num_workers GREATER 8)
            set(num_workers 8)
        endif()
    endif()

    message(VERBOSE "Generating individual contraction multi-ABD kernels for ${datatype} ${layout}...")

    file(MAKE_DIRECTORY ${working_path})

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/contraction_multi_abd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --a_elementwise ${CONTRACTION_MULTI_ABD_A_ELEMENTWISE}
                --b_elementwise ${CONTRACTION_MULTI_ABD_B_ELEMENTWISE}
                --cde_elementwise ${CONTRACTION_MULTI_ABD_CDE_ELEMENTWISE}
                --num_a_tensor ${CONTRACTION_MULTI_ABD_NUM_A_TENSOR}
                --num_b_tensor ${CONTRACTION_MULTI_ABD_NUM_B_TENSOR}
                --num_d_tensor ${CONTRACTION_MULTI_ABD_NUM_D_TENSOR}
                --num_dim_g ${CONTRACTION_MULTI_ABD_NUM_DIM_G}
                --num_dim_m ${CONTRACTION_MULTI_ABD_NUM_DIM_M}
                --num_dim_n ${CONTRACTION_MULTI_ABD_NUM_DIM_N}
                --num_dim_k ${CONTRACTION_MULTI_ABD_NUM_DIM_K}
                --config_json ${json_blob}
                --gpu_target ${CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    if(EXISTS ${working_path}/contraction_multi_abd_kernel_count.txt)
        file(READ ${working_path}/contraction_multi_abd_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    if(EXISTS ${working_path}/contraction_multi_abd_kernel_list.txt)
        file(STRINGS ${working_path}/contraction_multi_abd_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_contraction_multi_abd_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

message(VERBOSE "=== Starting Tile Engine Contraction Multi ABD Configuration ===")
message(VERBOSE "CONTRACTION_MULTI_ABD_DATATYPE: ${CONTRACTION_MULTI_ABD_DATATYPE}")
message(VERBOSE "CONTRACTION_MULTI_ABD_LAYOUT: ${CONTRACTION_MULTI_ABD_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

set(CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx942;gfx950;gfx12-generic")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

if(NOT CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine Contraction Multi ABD build: No supported GPU targets (gfx942, gfx950, gfx12-generic) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual Contraction Multi ABD targets for GPU targets: ${CONTRACTION_MULTI_ABD_GPU_TARGETS_INDIVIDUAL}")

    set(_PREV_CXX_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}")
    set(_CCACHE_APPLIED FALSE)

    if(ENABLE_CCACHE_CONTRACTION_MULTI_ABD)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            set(_CCACHE_APPLIED TRUE)
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    endif()

    add_custom_target(benchmark_contraction_multi_abd_all)

    foreach(dt IN LISTS CONTRACTION_MULTI_ABD_DATATYPE)
        add_custom_target(benchmark_contraction_multi_abd_${dt})
    endforeach()

    foreach(l IN LISTS CONTRACTION_MULTI_ABD_LAYOUT)
        add_custom_target(benchmark_contraction_multi_abd_${l})
    endforeach()

    foreach(dt IN LISTS CONTRACTION_MULTI_ABD_DATATYPE)
        foreach(l IN LISTS CONTRACTION_MULTI_ABD_LAYOUT)
            add_custom_target(benchmark_contraction_multi_abd_${dt}_${l})
        endforeach()
    endforeach()

    set(CONTRACTION_MULTI_ABD_PIPELINES "mem;compv3;compv4")
    set(CONTRACTION_MULTI_ABD_EPILOGUES "default;cshuffle")
    set(CONTRACTION_MULTI_ABD_SCHEDULERS "intrawave;interwave")

    foreach(pipeline IN LISTS CONTRACTION_MULTI_ABD_PIPELINES)
        add_custom_target(benchmark_contraction_multi_abd_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS CONTRACTION_MULTI_ABD_EPILOGUES)
        add_custom_target(benchmark_contraction_multi_abd_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS CONTRACTION_MULTI_ABD_SCHEDULERS)
        add_custom_target(benchmark_contraction_multi_abd_${scheduler}_scheduler)
    endforeach()

    foreach(dt IN LISTS CONTRACTION_MULTI_ABD_DATATYPE)
        foreach(l IN LISTS CONTRACTION_MULTI_ABD_LAYOUT)
            build_individual_contraction_multi_abd_targets(${dt} ${l})
        endforeach()
    endforeach()

    if(_CCACHE_APPLIED)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${_PREV_CXX_LAUNCHER}")
    endif()
    unset(_PREV_CXX_LAUNCHER)
    unset(_CCACHE_APPLIED)
endif()
