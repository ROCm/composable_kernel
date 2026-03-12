# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CU_COUNT 0 CACHE STRING "Number of Compute Units on the device")

# ============================================================================
# get_cu_count
#
# Returns the CU count for the device. If the given cu_count_arg is a positive
# integer, then the nothing happens. Otherwise, we attempt to query the CU
# count from the device. If the query is unsucessful, the default value of 100
# is returned.
#
# Parameters:
#   cu_count_arg  - The starting CU count
# ============================================================================
function(get_cu_count cu_count_arg)
    message(STATUS "Starting query for CU count needed for Stream-K test config generation")

    if(NOT "${${cu_count_arg}}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "The CU count must be a non-negative integer. \
                The given value of ${${cu_count_arg}} is invalid.")
    endif()

    if("${${cu_count_arg}}" STREQUAL "0")

        set(CPP_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cu_count.cpp)
        set(CPP_EXE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cu_count)

        execute_process(
            COMMAND ${CMAKE_HIP_COMPILER} -x hip ${CPP_FILE_PATH} -o ${CPP_EXE_PATH}
            RESULT_VARIABLE compile_exit_code
        )
        
        if (NOT compile_exit_code EQUAL 0)
            message(FATAL_ERROR "Compilation of ${CPP_FILE_PATH} failed.\n")
        endif()

        # Get the HIP library directory
        get_filename_component(HIP_COMPILER_DIR ${CMAKE_HIP_COMPILER} DIRECTORY)
        get_filename_component(HIP_ROOT_DIR ${HIP_COMPILER_DIR} DIRECTORY)
        set(HIP_LIB_DIR "${HIP_ROOT_DIR}/lib")

        # Set library path for runtime execution
        if(WIN32)
            set(ENV{PATH} "${HIP_LIB_DIR};$ENV{PATH}")
        else()
            set(ENV{LD_LIBRARY_PATH} "${HIP_LIB_DIR}:$ENV{LD_LIBRARY_PATH}")
        endif()

        execute_process(
            COMMAND ${CPP_EXE_PATH}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE standard_error
            OUTPUT_VARIABLE queried_cu_count
            RESULT_VARIABLE queried_cu_count_exit_code
        )

        if (standard_error)
            message(STATUS "Error information from attempting to query HIP device and properties:\n"
                            "${standard_error}")
        endif()

        if (NOT queried_cu_count_exit_code EQUAL 0)
            message(STATUS "Failed to run ${CPP_EXE_PATH} to query the device's CU count")

        endif()
        

        # Delete the generated cu_count executable
        file(REMOVE "${CPP_EXE_PATH}")

        if((queried_cu_count STREQUAL "0") OR (NOT queried_cu_count_exit_code EQUAL 0))
            message(WARNING "Unable to query the number of Compute Units. \
                    Please use the CU_COUNT CLI option to pass in the \
                    number of Compute Units for your target device; otherwise, \
                    the default value of 100 will be used.")
            set(${cu_count_arg} 100 PARENT_SCOPE)
        else()
            set(${cu_count_arg} ${queried_cu_count} PARENT_SCOPE)
        endif()

    endif()

endfunction()

# ============================================================================
# generate_test_configs
#
# Generate config json files for Stream-K tests
#
# Parameters:
#   cu_count_arg  - The number of CUs on the device
#   tile_sizes    - A list of block tile sizes: tile_m,tile_n,tile_k
#   datatype      - The datatype for which the config is being generated
#   config_list   - The variable to which the list of config file names are written
#   configs_path   - Path to the configs directory to which config files are written
# ============================================================================
function(generate_test_configs cu_count_arg tile_sizes datatype config_list configs_path)
    message(STATUS "Generating Stream-K test config files for ${datatype}")

    file(MAKE_DIRECTORY ${configs_path})

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/generate_configs.py 
                --cu_count ${cu_count_arg} 
                --configs_dir_path ${configs_path}
                --tiles ${tile_sizes}
                --datatype ${datatype}
        OUTPUT_VARIABLE CONFIG_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE script_ret_val
    )

    if (NOT script_ret_val EQUAL 0)
        message(FATAL_ERROR "Eror occured during execution of ${CMAKE_CURRENT_SOURCE_DIR}/generate_configs.py")
    endif()

    set(${config_list} ${CONFIG_LIST} PARENT_SCOPE)

endfunction()
