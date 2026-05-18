# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set(TEST_DATA_TYPE_COMPILE_OPTIONS)
if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_pk_int4 test_pk_int4.cpp)
endif()
if(GPU_TARGETS MATCHES "gfx95|gfx125")
    add_gtest_executable(test_ck_tile_pk_fp4 test_pk_fp4.cpp)
    if(result EQUAL 0)
      target_compile_options(test_ck_tile_pk_fp4 PRIVATE -mavx)
    endif()
    add_gtest_executable(test_ck_tile_mx_scale test_mx_scale.cpp)
    if(result EQUAL 0)
      target_compile_options(test_ck_tile_mx_scale PRIVATE -mavx)
    endif()
    add_gtest_executable(test_ck_tile_tf32 test_tf32.cpp)
    add_gtest_executable(test_ck_tile_bf16_f32_convert test_bf16_f32_convert.cpp)
endif()
if(GPU_TARGETS MATCHES "gfx125")
    # Check if host CPU supports AVX-512
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    check_cxx_source_runs("
        #include <immintrin.h>
        #include <cstring>
        int main() {
            // Create AVX-512 register and force actual execution
            __m512 a = _mm512_set1_ps(1.0f);
            __m512 b = _mm512_set1_ps(2.0f);
            __m512 c = _mm512_add_ps(a, b);  // Actual AVX-512 operation
            float result[16];
            _mm512_storeu_ps(result, c);     // Store to memory to prevent optimization
            // Use volatile to ensure the code isn't optimized away
            volatile float check = result[0];
            return (check == 3.0f) ? 0 : 1;
        }
    " HOST_HAS_AVX512F)
    set(CMAKE_REQUIRED_FLAGS "")
    
    add_gtest_executable(test_ck_tile_pk_fp6 test_pk_fp6.cpp)
    if(result EQUAL 0)
      if(HOST_HAS_AVX512F)
        message(STATUS "Host CPU supports AVX-512F, enabling -mavx512f for test_ck_tile_pk_fp6")
        target_compile_options(test_ck_tile_pk_fp6 PRIVATE -mavx512f)
        list(APPEND TEST_DATA_TYPE_COMPILE_OPTIONS -DCK_TILE_HOST_HAS_AVX512F)
      else()
        message(STATUS "Host CPU does not support AVX-512F, using -mavx2")
        target_compile_options(test_ck_tile_pk_fp6 PRIVATE -mavx2)
      endif()
      target_compile_options(test_ck_tile_pk_fp6 PRIVATE ${TEST_DATA_TYPE_COMPILE_OPTIONS})
    endif()
endif()
if(CK_USE_OCP_FP8 OR CK_USE_FNUZ_FP8)
    add_gtest_executable(test_ck_tile_fp8 test_fp8.cpp)
    target_compile_options(test_ck_tile_fp8 PRIVATE -Wno-float-equal)
    target_compile_definitions(test_ck_tile_fp8 PUBLIC GTEST_HAS_RTTI=0)
    # conditionally specify the use of OCP_FP8
    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()

# BF16 tests - modular approach
if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    # Consolidated BF16 tests (all tests in one file for consistency with other data type tests)
    add_gtest_executable(test_ck_tile_bf16 test_bf16.cpp)
    target_compile_options(test_ck_tile_bf16 PRIVATE -Wno-float-equal)

    # Apply common compile definitions based on build configuration
    set(BF16_TEST_TARGETS
        test_ck_tile_bf16
    )
    
    foreach(test_target ${BF16_TEST_TARGETS})

        # Apply LLVM builtin BF16 flag if explicitly set as a CMake variable.
        # If not set here, config.hpp will automatically define CK_TILE_USE_LLVM_BUILTIN_BF16 based on HIP version (ROCm 6.5.50421+ or ROCm 7.0+)
        # This allows the user to explicitly set the flag at CMake configure time (e.g., via -DCK_TILE_USE_LLVM_BUILTIN_BF16=ON/OFF as a CMake variable),
        # which is then converted to 0/1 for the C++ preprocessor, allowing them to override the header's default.
        if(DEFINED CK_TILE_USE_LLVM_BUILTIN_BF16)
            # Convert ON/OFF to 0/1 for C++ preprocessor
            if(CK_TILE_USE_LLVM_BUILTIN_BF16)
                target_compile_definitions(${test_target} PRIVATE CK_TILE_USE_LLVM_BUILTIN_BF16=1)
            else()
                target_compile_definitions(${test_target} PRIVATE CK_TILE_USE_LLVM_BUILTIN_BF16=0)
            endif()
        endif()

        # Define macro to indicate hardware/software saturation behavior for bf16 overflow
        # - gfx9 (gfx90a, gfx908, gfx942): Hardware saturates to bf16::max
        # - gfx11: Software truncate mode saturates to bf16::max
        # - gfx12: Hardware builtin saturates to bf16::max
        # - gfx950: Hardware builtin rounds to infinity (IEEE-754 RTN)
        if(GPU_TARGETS MATCHES "gfx90a|gfx908|gfx94[02]|gfx11")
            target_compile_definitions(${test_target} PRIVATE CK_TILE_BF16_OVERFLOW_SATURATES=1)
        endif()

        # Apply custom data type flag if enabled (except for arithmetic which already has it)
        if(CK_TILE_USE_CUSTOM_DATA_TYPE AND NOT "${test_target}" STREQUAL "test_ck_tile_bf16_arithmetic")
            target_compile_definitions(${test_target} PRIVATE CK_TILE_USE_CUSTOM_DATA_TYPE=1)
        endif()

        # Ensure GTEST_HAS_RTTI is consistent
        target_compile_definitions(${test_target} PUBLIC GTEST_HAS_RTTI=0)
    endforeach()
endif()
