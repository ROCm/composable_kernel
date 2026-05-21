# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Budget allocation when TILE_ENGINE_SAMPLING_TIER is set
if(NOT "${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "")
    # Map tier name to budget
    if("${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "daily")
        set(_te_budget 8000)
    elseif("${TILE_ENGINE_SAMPLING_TIER}" STREQUAL "weekly")
        set(_te_budget 0)
    else()
        set(_te_budget ${TILE_ENGINE_SAMPLING_TIER})
    endif()

    if(_te_budget GREATER 0)
        # Detect active ops from their DATATYPE variables
        set(_active_ops "")
        foreach(_op gemm_universal gemm_multi_d gemm_preshuffle grouped_gemm)
            string(TOUPPER ${_op} _OP_UPPER)
            if(NOT "${${_OP_UPPER}_DATATYPE}" STREQUAL "")
                list(APPEND _active_ops ${_op})
            endif()
        endforeach()

        if(_active_ops)
            string(REPLACE ";" "," _active_ops_csv "${_active_ops}")
            set(_alloc_dir "${CMAKE_CURRENT_BINARY_DIR}/sampling_alloc")
            file(MAKE_DIRECTORY ${_alloc_dir})

            execute_process(
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}/../../"
                    ${Python3_EXECUTABLE} -m sampling.allocate_budget
                    --total-budget ${_te_budget}
                    --active-ops "${_active_ops_csv}"
                    --output-dir ${_alloc_dir}
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../
                RESULT_VARIABLE _alloc_ret
                OUTPUT_VARIABLE _alloc_output
                ERROR_VARIABLE _alloc_error
            )
            if(NOT _alloc_ret EQUAL 0)
                message(FATAL_ERROR "Budget allocation failed: ${_alloc_error}")
            endif()
            message(STATUS "Sampling budget allocation:\n${_alloc_output}")

            # Read per-op allocations (only if not already overridden)
            foreach(_op gemm_universal gemm_multi_d gemm_preshuffle grouped_gemm)
                string(TOUPPER ${_op} _OP_UPPER)
                if("${${_OP_UPPER}_MAX_INSTANCES}" STREQUAL "")
                    if(EXISTS "${_alloc_dir}/${_op}_budget.txt")
                        file(READ "${_alloc_dir}/${_op}_budget.txt" _budget_val)
                        string(STRIP "${_budget_val}" _budget_val)
                        set(${_OP_UPPER}_MAX_INSTANCES ${_budget_val})
                        message(STATUS "  ${_op}: ${_budget_val} instances (from budget allocation)")
                    endif()
                else()
                    message(STATUS "  ${_op}: ${${_OP_UPPER}_MAX_INSTANCES} instances (explicit override)")
                endif()
            endforeach()
        endif()
    endif()
endif()

add_subdirectory(gemm_universal EXCLUDE_FROM_ALL)
add_subdirectory(gemm_multi_d EXCLUDE_FROM_ALL)
add_subdirectory(gemm_preshuffle EXCLUDE_FROM_ALL)
add_subdirectory(grouped_gemm EXCLUDE_FROM_ALL)
