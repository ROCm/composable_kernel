# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Resource check for the comp_tdm_v3 kernels of test_ck_tile_gemm_pipeline_comp_tdm_v3_wmma.
# comp_tdm_v3 is laid out to fit 256 VGPRs at 16 waves without spilling; a compiler change that
# makes it spill would still pass the numerical test, so fail here instead.
#
# Extracts the gfx125x code objects from the test binary (llvm-objdump --offloading), reads the
# AMDGPU metadata (llvm-readelf --notes) and requires, for every kernel whose name contains
# CompTDMV3, .private_segment_fixed_size 0 and .vgpr_spill_count 0. At least one such kernel must
# be found. Kernels whose name contains TensorWaitIssuersOnly or RefillAfterArrive (opt-in
# variants of the split barrier that spill with some compilers) are not checked.
#
# Usage: cmake -DTEST_BINARY=<exe> -DLLVM_OBJDUMP=<llvm-objdump> -DLLVM_READELF=<llvm-readelf>
#              -DWORK_DIR=<dir> -P check_comp_tdm_v3_resources.cmake

foreach(var TEST_BINARY LLVM_OBJDUMP LLVM_READELF WORK_DIR)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "check_comp_tdm_v3_resources: ${var} is not set")
    endif()
endforeach()

# llvm-objdump --offloading writes the extracted images next to its input, so work on a copy.
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
get_filename_component(bin_name "${TEST_BINARY}" NAME)
file(COPY "${TEST_BINARY}" DESTINATION "${WORK_DIR}")
execute_process(COMMAND "${LLVM_OBJDUMP}" --offloading "${WORK_DIR}/${bin_name}"
                WORKING_DIRECTORY "${WORK_DIR}"
                RESULT_VARIABLE rc
                OUTPUT_QUIET)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "llvm-objdump --offloading failed (${rc}) on ${TEST_BINARY}")
endif()

file(GLOB code_objects "${WORK_DIR}/${bin_name}.*gfx125*")
if(NOT code_objects)
    message(FATAL_ERROR "no gfx125x code object found in ${TEST_BINARY}")
endif()

set(num_kernels 0)
set(failures "")
foreach(co ${code_objects})
    execute_process(COMMAND "${LLVM_READELF}" --notes "${co}"
                    RESULT_VARIABLE rc
                    OUTPUT_VARIABLE notes)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "llvm-readelf --notes failed (${rc}) on ${co}")
    endif()
    string(REPLACE "\n" ";" lines "${notes}")
    # The metadata keys of a kernel are sorted, so .private_segment_fixed_size and
    # .vgpr_spill_count follow that kernel's .name.
    set(name "")
    foreach(line ${lines})
        if(line MATCHES "^[ ]+\\.name:[ ]+([^ ]+)")
            set(name "${CMAKE_MATCH_1}")
            if(name MATCHES "TensorWaitIssuersOnly|RefillAfterArrive")
                set(name "")
            endif()
            if(name MATCHES "CompTDMV3")
                math(EXPR num_kernels "${num_kernels} + 1")
            endif()
        elseif(name MATCHES "CompTDMV3" AND
               line MATCHES "^[ ]+\\.(private_segment_fixed_size|vgpr_spill_count):[ ]+([0-9]+)")
            if(NOT CMAKE_MATCH_2 EQUAL 0)
                list(APPEND failures "${CMAKE_MATCH_1} = ${CMAKE_MATCH_2} in ${name}")
            endif()
        endif()
    endforeach()
endforeach()

if(num_kernels EQUAL 0)
    message(FATAL_ERROR "no CompTDMV3 kernel found in ${TEST_BINARY}")
endif()
if(failures)
    string(REPLACE ";" "\n" failures "${failures}")
    message(FATAL_ERROR "comp_tdm_v3 kernels use scratch:\n${failures}")
endif()
message(STATUS "comp_tdm_v3 resources: ${num_kernels} kernels, no scratch, no VGPR spills")
