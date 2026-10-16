# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TDM (tensor data mover) batched contraction pipelines exist on gfx125x only.
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_batched_contraction_tdm_wmma
        test_batched_contraction_tdm_wmma.cpp)
    if(TARGET test_ck_tile_batched_contraction_tdm_wmma)
        ck_tile_restrict_archs(test_ck_tile_batched_contraction_tdm_wmma "^gfx125")
    endif()

    # Negative compile checks: each configuration must be rejected by a static_assert.
    # The object libraries are excluded from "all"; a ctest entry builds each one and
    # passes only when the build output contains the full text of the intended static_assert.
    # Entries are "<macro>|<regex>"; each regex is the complete, unique assert message.
    set(TDM_CONTRACTION_COMPILE_FAIL_CASES
        "NUM_D|TDM batched contraction does not support D tensors"
        "PAD_N|TDM batched contraction requires kPadK == false and kPadN == false"
        "MULTI_ABD|TDM batched contraction multi-ABD: TdmEpilogue overwrites E, so multi-ABD accumulation is unsupported"
        "CSHUFFLE_EPILOGUE|TDM batched contraction requires TdmEpilogue")

    # Compile only for the gfx125x targets that were requested.
    set(TDM_CONTRACTION_ARCHS ${SUPPORTED_GPU_TARGETS})
    list(FILTER TDM_CONTRACTION_ARCHS INCLUDE REGEX "gfx125")

    # --config is only meaningful (and only emitted) for multi-config generators.
    get_property(TDM_CONTRACTION_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    set_source_files_properties(test_batched_contraction_tdm_compile_fail.cpp
        PROPERTIES LANGUAGE HIP)
    foreach(fail_case IN LISTS TDM_CONTRACTION_COMPILE_FAIL_CASES)
        string(REPLACE "|" ";" fail_case_parts "${fail_case}")
        list(GET fail_case_parts 0 fail_macro)
        list(GET fail_case_parts 1 fail_regex)
        string(TOLOWER "${fail_macro}" fail_suffix)
        set(fail_target test_ck_tile_batched_contraction_tdm_compile_fail_${fail_suffix})
        add_library(${fail_target} OBJECT EXCLUDE_FROM_ALL
            test_batched_contraction_tdm_compile_fail.cpp)
        set_property(TARGET ${fail_target} PROPERTY HIP_ARCHITECTURES ${TDM_CONTRACTION_ARCHS})
        target_compile_definitions(${fail_target} PRIVATE CK_TDM_CONTRACTION_FAIL_${fail_macro})
        if(TDM_CONTRACTION_MULTI_CONFIG)
            add_test(NAME ${fail_target}
                COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${fail_target}
                        --config $<CONFIG>)
        else()
            add_test(NAME ${fail_target}
                COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${fail_target})
        endif()
        # RESOURCE_LOCK: these tests drive the shared build tree and must not run concurrently.
        set_tests_properties(${fail_target} PROPERTIES
            PASS_REGULAR_EXPRESSION "${fail_regex}"
            RESOURCE_LOCK ck_tile_build_tree
            LABELS "SMOKE_TEST;CK_TILE_BATCHED_CONTRACTION_TESTS")
    endforeach()
endif()
