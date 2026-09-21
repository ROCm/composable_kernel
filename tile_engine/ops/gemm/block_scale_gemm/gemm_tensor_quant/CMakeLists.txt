# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_TENSOR_QUANT_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for GEMM Tensor Quant (semicolon-separated)")
set(GEMM_TENSOR_QUANT_LAYOUT "rcr" CACHE STRING "List of layouts for GEMM Tensor Quant (semicolon-separated)")
set(GEMM_TENSOR_QUANT_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(GEMM_TENSOR_QUANT_MAX_INSTANCES "" CACHE STRING "Max kernel instances per (dtype, layout) combo (empty = no cap)")
option(ENABLE_CCACHE_GEMM_TENSOR_QUANT "Enable ccache for GEMM Tensor Quant ops compilation" OFF)

set(GEMM_TENSOR_QUANT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(create_individual_gemm_tensor_quant_target datatype layout kernel_name trait tile_config config_json)
    set(target_name "benchmark_${kernel_name}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    string(REPLACE "gemm_tensor_quant_" "" simplified_name ${kernel_name})
    set(instance_header "${working_path}/gemm_tensor_quant_single_${simplified_name}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_TENSOR_QUANT_SOURCE_DIR}/gemm_tensor_quant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "${kernel_name}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_TENSOR_QUANT_SOURCE_DIR}/gemm_tensor_quant_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_TENSOR_QUANT_SOURCE_DIR}/gemm_tensor_quant_benchmark_single.cpp
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL})

    target_include_directories(${target_name} PRIVATE
        ${GEMM_TENSOR_QUANT_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    add_dependencies(benchmark_gemm_tensor_quant_all ${target_name})
    add_dependencies(benchmark_gemm_tensor_quant_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_tensor_quant_${layout} ${target_name})
    add_dependencies(benchmark_gemm_tensor_quant_${datatype}_${layout} ${target_name})
endfunction()

function(build_individual_gemm_tensor_quant_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if(DEFINED ENV{GEMM_TENSOR_QUANT_CONFIG_FILE} AND NOT "$ENV{GEMM_TENSOR_QUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/$ENV{GEMM_TENSOR_QUANT_CONFIG_FILE}")
    elseif(NOT "${GEMM_TENSOR_QUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_TENSOR_QUANT_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    file(MAKE_DIRECTORY ${working_path})

    # Build optional args for instance builder
    set(extra_list_args "")
    if(NOT "${GEMM_TENSOR_QUANT_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${GEMM_TENSOR_QUANT_MAX_INSTANCES})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
        list(APPEND extra_list_args --tier ${TILE_ENGINE_SAMPLING_TIER})
        list(APPEND extra_list_args --manifest-path ${working_path})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_SEED}" STREQUAL "")
        list(APPEND extra_list_args --seed ${TILE_ENGINE_SAMPLING_SEED})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_tensor_quant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target ${GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL}
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

    if(EXISTS ${working_path}/gemm_tensor_quant_kernel_count.txt)
        file(READ ${working_path}/gemm_tensor_quant_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found for GEMM Tensor Quant")
    endif()

    if(EXISTS ${working_path}/gemm_tensor_quant_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_tensor_quant_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_tensor_quant_target(
                "${datatype}" "${layout}" "${kernel_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found for GEMM Tensor Quant")
    endif()
endfunction()

set(GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201;gfx12-generic")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL ${target})
    endif()
endforeach()

if(NOT GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM Tensor Quant build: No supported GPU targets found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM Tensor Quant targets for GPU targets: ${GEMM_TENSOR_QUANT_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if requested
    if(ENABLE_CCACHE_GEMM_TENSOR_QUANT)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        endif()
    endif()

    add_custom_target(benchmark_gemm_tensor_quant_all)

    foreach(datatype ${GEMM_TENSOR_QUANT_DATATYPE})
        add_custom_target(benchmark_gemm_tensor_quant_${datatype})
    endforeach()

    foreach(layout ${GEMM_TENSOR_QUANT_LAYOUT})
        add_custom_target(benchmark_gemm_tensor_quant_${layout})
    endforeach()

    foreach(datatype ${GEMM_TENSOR_QUANT_DATATYPE})
        foreach(layout ${GEMM_TENSOR_QUANT_LAYOUT})
            add_custom_target(benchmark_gemm_tensor_quant_${datatype}_${layout})
        endforeach()
    endforeach()

    # Divide MAX_INSTANCES budget across all active (dtype, layout) combos so that
    # sampling fires per-combo rather than being a single cap larger than any combo's
    # feasible set.
    if(NOT "${GEMM_TENSOR_QUANT_MAX_INSTANCES}" STREQUAL "")
        list(LENGTH GEMM_TENSOR_QUANT_DATATYPE _gtq_n_dt)
        list(LENGTH GEMM_TENSOR_QUANT_LAYOUT _gtq_n_lay)
        math(EXPR _gtq_n_combos "${_gtq_n_dt} * ${_gtq_n_lay}")
        if(_gtq_n_combos GREATER 0)
            math(EXPR GEMM_TENSOR_QUANT_MAX_INSTANCES
                "${GEMM_TENSOR_QUANT_MAX_INSTANCES} / ${_gtq_n_combos}")
            message(STATUS "  gemm_tensor_quant: per-combo budget = ${GEMM_TENSOR_QUANT_MAX_INSTANCES} (${_gtq_n_combos} combos)")
        endif()
    endif()

    foreach(datatype ${GEMM_TENSOR_QUANT_DATATYPE})
        foreach(layout ${GEMM_TENSOR_QUANT_LAYOUT})
            build_individual_gemm_tensor_quant_targets(${datatype} ${layout})
        endforeach()
    endforeach()
endif()
