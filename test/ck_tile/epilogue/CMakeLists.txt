# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp16 test_cshuffle_epilogue_fp16.cpp)
add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp8 test_cshuffle_epilogue_fp8.cpp)
add_gtest_executable(test_ck_tile_cshuffle_epilogue_scale test_cshuffle_epilogue_scale.cpp)

if(CK_USE_OCP_FP8)
    target_compile_options(test_ck_tile_cshuffle_epilogue_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp8_gfx950 test_cshuffle_epilogue_fp8_gfx950.cpp)
    add_gtest_executable(test_ck_tile_permuten_epilogue_scale test_permuten_epilogue_scale.cpp)
    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_cshuffle_epilogue_fp8_gfx950 PRIVATE -DCK_TILE_USE_OCP_FP8)
        target_compile_options(test_ck_tile_permuten_epilogue_scale PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()

add_gtest_executable(test_ck_tile_lds_shuffle_2d_epilogue test_lds_shuffle_2d_epilogue.cpp)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_tdm_epilogue test_tdm_epilogue.cpp)
    if(result EQUAL 0)
        # TDM is gfx1250-only; skip device compilation for the other targets of a mixed build.
        get_target_property(TDM_EPILOGUE_TARGETS test_ck_tile_tdm_epilogue HIP_ARCHITECTURES)
        list(FILTER TDM_EPILOGUE_TARGETS INCLUDE REGEX "^gfx125")
        set_property(TARGET test_ck_tile_tdm_epilogue PROPERTY HIP_ARCHITECTURES ${TDM_EPILOGUE_TARGETS})
        # A TDM store that never retires hangs in s_wait_tensorcnt instead of failing.
        set_tests_properties(test_ck_tile_tdm_epilogue PROPERTIES TIMEOUT 120)
    endif()
endif()

# Exercise both epilogue callers of buffer_store_fence with gfx12's separate store counter.
if(GPU_TARGETS MATCHES "gfx12")
    set(GFX12_EPILOGUE_TARGETS ${SUPPORTED_GPU_TARGETS})
    list(FILTER GFX12_EPILOGUE_TARGETS INCLUDE REGEX "^gfx12")
    foreach(use_builtin IN ITEMS 1 0)
        set(test_target test_ck_tile_buffer_store_fence_epilogues)
        if(NOT use_builtin)
            string(APPEND test_target "_legacy")
        endif()
        add_gtest_executable(${test_target} test_buffer_store_fence_epilogues.cpp)
        set_property(TARGET ${test_target}
                     PROPERTY HIP_ARCHITECTURES ${GFX12_EPILOGUE_TARGETS})
        target_compile_definitions(${test_target} PRIVATE
            CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN=${use_builtin})
    endforeach()
endif()
