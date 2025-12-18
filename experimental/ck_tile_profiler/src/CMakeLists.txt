# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
if (CK_EXPERIMENTAL_PROFILER)

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances STATIC
        tile_grouped_conv_bwd_data_bf16_instances.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_2 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_2.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_3 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_3.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_4 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_4.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_5 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_5.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_6 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_6.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_data_bf16_instances_7 STATIC
        tile_grouped_conv_bwd_data_bf16_instances_7.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_weight_fp16_instances STATIC
        tile_grouped_conv_bwd_weight_fp16_instances.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_weight_bf16_instances STATIC
        tile_grouped_conv_bwd_weight_bf16_instances.cpp
    )

    add_library(ck_tile_grouped_conv_bwd_weight_bf16_instances_opt STATIC
        tile_grouped_conv_bwd_weight_bf16_instances_opt.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_fp16_instances STATIC
        tile_grouped_conv_fwd_fp16_instances.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances STATIC
        tile_grouped_conv_fwd_bf16_instances.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances_2 STATIC
        tile_grouped_conv_fwd_bf16_instances_2.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances_3 STATIC
        tile_grouped_conv_fwd_bf16_instances_3.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances_4 STATIC
        tile_grouped_conv_fwd_bf16_instances_4.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances_5 STATIC
        tile_grouped_conv_fwd_bf16_instances_5.cpp
    )

    add_library(ck_tile_grouped_conv_fwd_bf16_instances_6 STATIC
        tile_grouped_conv_fwd_bf16_instances_6.cpp
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_2 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_3 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_4 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_5 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_6 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_data_bf16_instances_7 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_weight_fp16_instances PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_weight_bf16_instances PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_bwd_weight_bf16_instances_opt PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_fp16_instances PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_2 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_3 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_4 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_5 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )

    target_include_directories(ck_tile_grouped_conv_fwd_bf16_instances_6 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
        ${PROJECT_SOURCE_DIR}/experimental/ck_tile_profiler/include
    )


    set(CK_PROFILER_OP_FILTER "" CACHE STRING "Filter for the operators to be profiled. Default is to include all")
    set(CK_PROFILER_INSTANCE_FILTER "" CACHE STRING "Filter for the kernels instances to be profiled. Default is to be the same as the operator filter")
    if (CK_PROFILER_OP_FILTER STREQUAL "")
    set(CK_PROFILER_OP_FILTER ".+")
    endif()
    if (CK_PROFILER_INSTANCE_FILTER STREQUAL "")
    set(CK_PROFILER_INSTANCE_FILTER ${CK_PROFILER_OP_FILTER})
    endif()
    message(STATUS "CK_PROFILER_OP_FILTER: ${CK_PROFILER_OP_FILTER}")
    message(STATUS "CK_PROFILER_INSTANCE_FILTER: ${CK_PROFILER_INSTANCE_FILTER}")

    if(SUPPORTED_GPU_TARGETS MATCHES "gfx9" OR SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]")
    list(APPEND PROFILER_OPS tile_profile_grouped_conv_bwd_data.cpp)
    list(APPEND PROFILER_OPS tile_profile_grouped_conv_bwd_weight.cpp)
    list(APPEND PROFILER_OPS tile_profile_grouped_conv_fwd.cpp)
    endif()

    set(PROFILER_SOURCES tile_profiler.cpp)
    foreach(SOURCE ${PROFILER_OPS})
    string(REGEX REPLACE "tile_profile_(.+)\.cpp" "\\1" OP_NAME ${SOURCE})
    if (OP_NAME STREQUAL "") 
        message(FATAL_ERROR "Unexpected source file name: ${SOURCE}")
    endif()
    if("${OP_NAME}" MATCHES "${CK_PROFILER_OP_FILTER}")
        list(APPEND PROFILER_SOURCES ${SOURCE})
    endif()
    endforeach()
    message(VERBOSE "ckTileProfiler sources: ${PROFILER_SOURCES}")

    set(PROFILER_EXECUTABLE ckTileProfiler)

    add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
    target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)
    # flags to compress the library
    if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
    message(DEBUG "Adding --offload-compress flag for ${PROFILER_EXECUTABLE}")
    target_compile_options(${PROFILER_EXECUTABLE} PRIVATE --offload-compress)
    endif()

    set(DEVICE_INSTANCES "")
    if(SUPPORTED_GPU_TARGETS MATCHES "gfx9" OR SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]")
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_2)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_3)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_4)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_5)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_6)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_data_bf16_instances_7)

    #list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_weight_fp16_instances)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_weight_bf16_instances)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_bwd_weight_bf16_instances_opt)
    #list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_fp16_instances)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances_2)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances_3)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances_4)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances_5)
    list(APPEND DEVICE_INSTANCES ck_tile_grouped_conv_fwd_bf16_instances_6)

    endif()

    if(SUPPORTED_GPU_TARGETS MATCHES "gfx9" OR SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]")
    endif()

    if(DL_KERNELS)
    endif()

    set(PROFILER_LIBS utility getopt::getopt)
    foreach(LIB ${DEVICE_INSTANCES})
    # Handle both traditional device_*_instance and ck_tile_*_instances patterns
    string(REGEX REPLACE "device_(.+)_instance" "\\1" INSTANCE_NAME ${LIB})
    if (INSTANCE_NAME STREQUAL ${LIB})
        # If no match, try ck_tile pattern
        string(REGEX REPLACE "ck_tile_(.+)_instances" "\\1" INSTANCE_NAME ${LIB})
    endif()
    if (INSTANCE_NAME STREQUAL "") 
        message(FATAL_ERROR "Unexpected kernel instance name: ${LIB}")
    endif()
    if("${INSTANCE_NAME}" MATCHES "${CK_PROFILER_INSTANCE_FILTER}")
        list(APPEND PROFILER_LIBS ${LIB})
    endif()
    endforeach()
    message(VERBOSE "ckTileProfiler libs: ${PROFILER_LIBS}")
    target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE ${PROFILER_LIBS})

    rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
endif()
