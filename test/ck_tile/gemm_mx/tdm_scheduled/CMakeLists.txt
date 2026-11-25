# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# comp_tdm_scheduled (the gfx1250 MX GEMM pipeline). Included from ../CMakeLists.txt under its gfx125
# block, so TEST_MX_GEMM_COMPILE_OPTIONS is set.
# Host-only: the LDS layout and policy rules, the K-major scale helpers and the reference.
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled_host test_mx_gemm_tdm_scheduled_host.cpp)
# The host test includes parity_cases.inc: the rows of parity_cases.py, which the dispatcher's
# test_mx_gemm_tdm_scheduled.py also checks against tile_engine's mx_tdm_scheduled.py.
if(TARGET test_ck_tile_mx_gemm_tdm_scheduled_host)
    set(MX_TDM_SCHEDULED_PARITY_INC ${CMAKE_CURRENT_BINARY_DIR}/parity_cases.inc)
    add_custom_command(
        OUTPUT ${MX_TDM_SCHEDULED_PARITY_INC}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/parity_cases.py
                --output ${MX_TDM_SCHEDULED_PARITY_INC}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/parity_cases.py
                ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/mx_gemm/mx_tdm_scheduled.py
        COMMENT "Generating comp_tdm_scheduled parity_cases.inc")
    target_sources(test_ck_tile_mx_gemm_tdm_scheduled_host PRIVATE ${MX_TDM_SCHEDULED_PARITY_INC})
    target_include_directories(test_ck_tile_mx_gemm_tdm_scheduled_host
                               PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled_host_ref test_mx_gemm_tdm_scheduled_host_ref.cpp)
# Every GPU case compiled for gfx1250 and checked as a problem its instance runs, without a launch.
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled_cases test_mx_gemm_tdm_scheduled.cpp)
# On a gfx1250 device: bitwise against the reference on the sparse check inputs. The smoke test
# runs a few small cases; the full and large-case tests are regression tests (test/CMakeLists.txt).
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled_smoke test_mx_gemm_tdm_scheduled_smoke.cpp)
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled test_mx_gemm_tdm_scheduled.cpp)
add_gtest_executable(test_ck_tile_mx_gemm_tdm_scheduled_large_cases test_mx_gemm_tdm_scheduled_large_cases.cpp)

# The scheduling options of example 56, so the tests check the code the example times: expert
# scheduling mode, no post-RA machine scheduler, and register class priority over globalness in
# the greedy allocator. The machine scheduler stays on, steered by the pipeline's hot loop hints;
# CK_TDM_SCHEDULED_SOURCE_ORDER turns it off and keeps the source order as the schedule.
option(CK_TDM_SCHEDULED_SOURCE_ORDER "comp_tdm_scheduled: source order schedule, misched off" OFF)
set(MX_TDM_SCHEDULED_SCHED_OPTIONS "SHELL:-mllvm -amdgpu-expert-scheduling-mode")
if(CK_TDM_SCHEDULED_SOURCE_ORDER)
    list(APPEND MX_TDM_SCHEDULED_SCHED_OPTIONS "SHELL:-mllvm -enable-misched=false"
                                               -DCK_TILE_TDM_SCHEDULED_SOURCE_ORDER=1)
endif()
list(APPEND MX_TDM_SCHEDULED_SCHED_OPTIONS "SHELL:-mllvm -enable-post-misched=false"
                                           "SHELL:-mllvm -greedy-regclass-priority-trumps-globalness")
foreach(test_target test_ck_tile_mx_gemm_tdm_scheduled_host test_ck_tile_mx_gemm_tdm_scheduled_host_ref
                    test_ck_tile_mx_gemm_tdm_scheduled_cases test_ck_tile_mx_gemm_tdm_scheduled_smoke
                    test_ck_tile_mx_gemm_tdm_scheduled test_ck_tile_mx_gemm_tdm_scheduled_large_cases)
    if(TARGET ${test_target})
        target_compile_options(${test_target} PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS}
                                                      ${MX_TDM_SCHEDULED_SCHED_OPTIONS})
        ck_tile_restrict_archs(${test_target} "^gfx125")
    endif()
endforeach()
if(TARGET test_ck_tile_mx_gemm_tdm_scheduled_cases)
    target_compile_definitions(test_ck_tile_mx_gemm_tdm_scheduled_cases PRIVATE MX_TDM_SCHEDULED_DRY_RUN)
endif()
