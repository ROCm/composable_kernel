# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Expected compilation failure tests for rocm_ck schema types.
#
# Each test compiles a .cpp file that must FAIL. PASS_REGULAR_EXPRESSION
# verifies the compiler emits the expected consteval throw message — not
# just any error. This catches tests that fail for the wrong reason
# (e.g., missing arguments instead of the intended validation error).
#
# To add a new test: add an add_compile_fail_test() call with the source
# file name (without .cpp) and a regex matching the expected error message.
#
# Usage:
#   ctest --test-dir build -R compile_fail
#   ctest --test-dir build -L ROCM_CK_COMPILE_FAIL

function(add_compile_fail_test test_name expected_error)
    set(target_name "compile_fail_${test_name}")
    set(ctest_name "rocm_ck_compile_fail_${test_name}")

    add_library(${target_name} OBJECT EXCLUDE_FROM_ALL ${test_name}.cpp)
    target_link_libraries(${target_name} PRIVATE rocm_ck)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_compile_options(${target_name} PRIVATE
        -Wno-zero-as-null-pointer-constant
    )

    add_test(
        NAME ${ctest_name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${target_name}
    )
    set_tests_properties(${ctest_name} PROPERTIES
        LABELS "ROCM_CK_COMPILE_FAIL"
        PASS_REGULAR_EXPRESSION "${expected_error}"
    )
endfunction()

# -- gemm_spec.hpp: tile geometry validation --
add_compile_fail_test(block_tile_indivisible   "block_tile\\.m must be divisible")
add_compile_fail_test(block_tile_n_indivisible "block_tile\\.n must be divisible")
add_compile_fail_test(block_tile_k_indivisible "block_tile\\.k must be divisible")
add_compile_fail_test(block_tile_negative      "block_tile dimensions must be positive")
add_compile_fail_test(block_waves_negative     "block_waves dimensions must be positive")
add_compile_fail_test(wave_tile_negative       "wave_tile dimensions must be positive")
add_compile_fail_test(invalid_wave_tile        "wave_tile is not a valid instruction shape")
add_compile_fail_test(block_waves_k_not_1      "CShuffle epilogue requires block_waves\\.k == 1")
add_compile_fail_test(k_batch_zero             "k_batch must be positive")

# -- gemm_spec.hpp: pipeline and partitioner constraints --
add_compile_fail_test(interwave_requires_memory "Interwave scheduling requires Pipeline::Memory")
add_compile_fail_test(streamk_with_splitk       "Stream-K tile partitioning is incompatible with split-K")
add_compile_fail_test(preshuffle_a_not_row      "Preshuffle pipeline requires A layout = Row")
add_compile_fail_test(preshuffle_b_not_col      "Preshuffle pipeline requires B layout = Col")

# -- gemm_spec.hpp: epilogue and store strategy --
add_compile_fail_test(direct2d_with_d_tensor       "Direct2D epilogue does not support D tensors")
add_compile_fail_test(too_many_epilogue_ops         "too many epilogue operations")
add_compile_fail_test(too_many_physical_tensors     "maximum 2 D tensors")
add_compile_fail_test(first_op_not_gemm             "GEMM makeSpec requires GemmOp as first operator")

# -- gemm_spec.hpp: dtype and target constraints --
add_compile_fail_test(fp8_wave_tile_wrong_target "wave_tile is not a valid instruction shape")
add_compile_fail_test(i4_without_quantize        "rhs dtype is I4 but Tensor\\.quantize is not set")
add_compile_fail_test(i8_on_gfx90a               "INT8 GEMM requires gfx942")
add_compile_fail_test(i8_pipeline_v1             "INT8 GEMM requires V3/V4/Memory pipeline")
add_compile_fail_test(i8_wrong_acc_dtype         "INT8 GEMM requires I32 accumulator")

# -- resolve.hpp --
add_compile_fail_test(conflicting_layout         "conflicting layout for tensor")
add_compile_fail_test(quantize_empty_scale_name  "quantize has empty scale_name")
add_compile_fail_test(no_dtype                   "tensor dtype unresolvable")
add_compile_fail_test(empty_tensor_name          "operator slot has empty tensor name")
add_compile_fail_test(ssa_violation              "tensor name produced by multiple operators")
add_compile_fail_test(scale_scalar_not_found     "ScaleOp.scale references undeclared Scalar")

# -- signature.hpp --
add_compile_fail_test(duplicate_scalar      "duplicate scalar name in Signature")
add_compile_fail_test(scale_missing_scalar  "ScaleOp\\.scale references undeclared Scalar")
add_compile_fail_test(tensor_no_name        "Tensor entry has metadata but no name")
add_compile_fail_test(too_many_ops          "excess elements in struct initializer")

# -- arch_properties.hpp --
add_compile_fail_test(target_set_mixed_wave       "wavefront_size.*requires all targets")
add_compile_fail_test(fp8_ocp_unsupported         "FP8_OCP.*not yet supported")
add_compile_fail_test(empty_target_set_wavefront  "wavefront_size.*called on empty")

# -- fixed_string.hpp --
add_compile_fail_test(tensor_name_too_long  "FixedString.*exceeds capacity")

# -- layout.hpp --
add_compile_fail_test(layout_auto_stride  "leadingDimStride requires Row or Col")
