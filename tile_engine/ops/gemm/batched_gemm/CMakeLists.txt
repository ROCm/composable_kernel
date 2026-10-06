# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(BATCHED_GEMM_DATATYPE "fp16" CACHE STRING "List of datatypes for batched GEMM (semicolon-separated)")
set(BATCHED_GEMM_LAYOUT "rcr" CACHE STRING "List of layouts for batched GEMM (semicolon-separated)")
set(BATCHED_GEMM_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(BATCHED_GEMM_MAX_INSTANCES "" CACHE STRING "Max kernel instances per (dtype, layout) combo (empty = no cap)")

set(BATCHED_GEMM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Trait-string splitter that understands multi-token pipelines (comp_tdm_v2)
include(${CMAKE_CURRENT_LIST_DIR}/../gemm_trait_parse.cmake)

function(create_individual_batched_gemm_target datatype layout trait tile_config config_json)
    if(NOT BATCHED_GEMM_GPU_TARGETS)
        message(WARNING "Skipping individual batched GEMM target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_batched_gemm_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    set(instance_header "${working_path}/batched_gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${BATCHED_GEMM_SOURCE_DIR}/batched_gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gpu_target "${BATCHED_GEMM_GPU_TARGETS}"
                --gen_single
                --kernel_name "batched_gemm_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
        DEPENDS ${BATCHED_GEMM_SOURCE_DIR}/batched_gemm_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${BATCHED_GEMM_SOURCE_DIR}/batched_gemm_benchmark_single.cpp
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${BATCHED_GEMM_GPU_TARGETS})

    target_compile_definitions(${target_name} PRIVATE
        BATCHED_GEMM_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    target_include_directories(${target_name} PRIVATE
        ${BATCHED_GEMM_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    add_dependencies(benchmark_batched_gemm_all ${target_name})
    add_dependencies(benchmark_batched_gemm_${datatype} ${target_name})
    add_dependencies(benchmark_batched_gemm_${layout} ${target_name})
    add_dependencies(benchmark_batched_gemm_${datatype}_${layout} ${target_name})

    ck_tile_gemm_split_trait("${trait}" pipeline epilogue scheduler)

    add_dependencies(benchmark_batched_gemm_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_batched_gemm_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_batched_gemm_${scheduler}_scheduler ${target_name})
endfunction()

function(build_individual_batched_gemm_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if(DEFINED ENV{BATCHED_GEMM_CONFIG_FILE} AND NOT "$ENV{BATCHED_GEMM_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{BATCHED_GEMM_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${BATCHED_GEMM_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${BATCHED_GEMM_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${BATCHED_GEMM_CONFIG_FILE}")
    elseif(BATCHED_GEMM_GPU_TARGETS MATCHES "gfx1250")
        # gfx1250 (WMMA) has its own warp-tile set and the TDM/async pipelines
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config_gfx1250.json")
        message(VERBOSE "  Using gfx1250 default config for batched GEMM")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for batched GEMM")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    file(MAKE_DIRECTORY ${working_path})

    # Build optional args for instance builder
    set(extra_list_args "")
    if(NOT "${BATCHED_GEMM_MAX_INSTANCES}" STREQUAL "")
        list(APPEND extra_list_args --max-instances ${BATCHED_GEMM_MAX_INSTANCES})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
        list(APPEND extra_list_args --tier ${TILE_ENGINE_SAMPLING_TIER})
        list(APPEND extra_list_args --manifest-path ${working_path})
    endif()
    if(NOT "${TILE_ENGINE_SAMPLING_SEED}" STREQUAL "")
        list(APPEND extra_list_args --seed ${TILE_ENGINE_SAMPLING_SEED})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${BATCHED_GEMM_SOURCE_DIR}/batched_gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target "${BATCHED_GEMM_GPU_TARGETS}"
                --list_kernels
                ${extra_list_args}
        WORKING_DIRECTORY ${BATCHED_GEMM_SOURCE_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    if(EXISTS ${working_path}/batched_gemm_kernel_count.txt)
        file(READ ${working_path}/batched_gemm_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    if(EXISTS ${working_path}/batched_gemm_kernel_list.txt)
        file(STRINGS ${working_path}/batched_gemm_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_batched_gemm_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found: ${working_path}/batched_gemm_kernel_list.txt")
    endif()
endfunction()

message(VERBOSE "=== Starting Tile Engine Batched GEMM Configuration ===")
message(VERBOSE "BATCHED_GEMM_DATATYPE: ${BATCHED_GEMM_DATATYPE}")
message(VERBOSE "BATCHED_GEMM_LAYOUT: ${BATCHED_GEMM_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

set(BATCHED_GEMM_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND BATCHED_GEMM_GPU_TARGETS ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# gfx1250 uses a different config (WMMA warp tiles, TDM/async pipelines) and a
# single --gpu_target per kernel, so it is only built when it is the sole
# matching target. In a mixed target list it is skipped and the other targets
# are configured exactly as before.
set(_BATCHED_GEMM_GFX1250_TARGETS "")
foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target MATCHES "^gfx1250(:|$)")
        list(APPEND _BATCHED_GEMM_GFX1250_TARGETS ${target})
    endif()
endforeach()
list(REMOVE_DUPLICATES _BATCHED_GEMM_GFX1250_TARGETS)
list(LENGTH _BATCHED_GEMM_GFX1250_TARGETS _bg_n_gfx1250)
if(_bg_n_gfx1250 GREATER 0)
    if(NOT BATCHED_GEMM_GPU_TARGETS AND _bg_n_gfx1250 EQUAL 1)
        set(BATCHED_GEMM_GPU_TARGETS ${_BATCHED_GEMM_GFX1250_TARGETS})
        message(VERBOSE "  Adding GPU target: ${_BATCHED_GEMM_GFX1250_TARGETS}")
    else()
        message(WARNING "Skipping gfx1250 for Tile Engine batched GEMM: gfx1250 must be the only GPU target (found ${SUPPORTED_GPU_TARGETS})")
    endif()
endif()

if(NOT BATCHED_GEMM_GPU_TARGETS)
    message(WARNING "Skipping Tile Engine batched GEMM build: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201, gfx1250) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual batched GEMM targets for GPU targets: ${BATCHED_GEMM_GPU_TARGETS}")

    add_custom_target(benchmark_batched_gemm_all)

    foreach(dt IN LISTS BATCHED_GEMM_DATATYPE)
        add_custom_target(benchmark_batched_gemm_${dt})
    endforeach()

    foreach(l IN LISTS BATCHED_GEMM_LAYOUT)
        add_custom_target(benchmark_batched_gemm_${l})
    endforeach()

    foreach(dt IN LISTS BATCHED_GEMM_DATATYPE)
        foreach(l IN LISTS BATCHED_GEMM_LAYOUT)
            add_custom_target(benchmark_batched_gemm_${dt}_${l})
        endforeach()
    endforeach()

    set(BATCHED_GEMM_PIPELINES "mem;compv3;compv4")
    set(BATCHED_GEMM_EPILOGUES "default;cshuffle")
    set(BATCHED_GEMM_SCHEDULERS "intrawave;interwave")
    if(BATCHED_GEMM_GPU_TARGETS MATCHES "gfx1250")
        # gfx1250 pipelines: non-MX comp_async (cshuffle epilogue) and the TDM
        # pipelines comp_tdm / comp_tdm_v2 (tdm epilogue). The weight-preshuffle
        # pipelines are not built: the batched kernel's per-batch B offset
        # assumes an unshuffled B.
        list(APPEND BATCHED_GEMM_PIPELINES "comp_async;comp_tdm;comp_tdm_v2")
        list(APPEND BATCHED_GEMM_EPILOGUES "tdm")
    endif()

    foreach(pipeline IN LISTS BATCHED_GEMM_PIPELINES)
        add_custom_target(benchmark_batched_gemm_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS BATCHED_GEMM_EPILOGUES)
        add_custom_target(benchmark_batched_gemm_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS BATCHED_GEMM_SCHEDULERS)
        add_custom_target(benchmark_batched_gemm_${scheduler}_scheduler)
    endforeach()

    foreach(dt IN LISTS BATCHED_GEMM_DATATYPE)
        foreach(l IN LISTS BATCHED_GEMM_LAYOUT)
            build_individual_batched_gemm_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
