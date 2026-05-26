// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/validate.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::AddOp;
using ::rocm_ck::Args;
using ::rocm_ck::DataType;
using ::rocm_ck::GemmAlgorithm;
using ::rocm_ck::GemmOp;
using ::rocm_ck::GemmSpec;
using ::rocm_ck::makeShape;
using ::rocm_ck::makeSpec;
using ::rocm_ck::makeStrides;
using ::rocm_ck::Signature;
using ::rocm_ck::TargetSet;
using ::rocm_ck::validate;

// A minimal GemmSpec for testing: 3 physical tensors (A, B, C).
static constexpr auto test_spec = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{
        .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {16, 16, 16}},
    TargetSet::cdna());

// A GemmSpec with a D0 tensor (4 physical tensors: A, B, D, C).
static constexpr auto test_spec_d0 =
    makeSpec(Signature{.dtype = DataType::FP16,
                       .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                 AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
             GemmAlgorithm{
                 .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {16, 16, 16}},
             TargetSet::cdna());

// ============================================================================
// validate() — passes when all tensors are filled
// ============================================================================

TEST(Validate, PassesWhenAllTensorsFilled)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3;

    Args args{};
    args.tensors[test_spec.lhs().args_slot]    = {&dummy_a, makeShape(64, 32), makeStrides(32, 1)};
    args.tensors[test_spec.rhs().args_slot]    = {&dummy_b, makeShape(32, 64), makeStrides(1, 32)};
    args.tensors[test_spec.output().args_slot] = {&dummy_c, makeShape(64, 64), makeStrides(64, 1)};

    // Should not abort
    validate(args, test_spec);
}

TEST(Validate, PassesWithD0TensorFilled)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3, dummy_bias = 4;

    Args args{};
    args.tensors[test_spec_d0.lhs().args_slot] = {&dummy_a, makeShape(64, 32), makeStrides(32, 1)};
    args.tensors[test_spec_d0.rhs().args_slot] = {&dummy_b, makeShape(32, 64), makeStrides(1, 32)};
    args.tensors[test_spec_d0.output().args_slot] = {
        &dummy_c, makeShape(64, 64), makeStrides(64, 1)};
    args.tensors[test_spec_d0.d0().args_slot] = {
        &dummy_bias, makeShape(64, 64), makeStrides(64, 1)};

    // Should not abort
    validate(args, test_spec_d0);
}

// ============================================================================
// validate() — aborts when a tensor is missing
// ============================================================================

#ifndef NDEBUG
TEST(ValidateDeathTest, AbortsOnNullTensorPointer)
{
    int dummy_a = 1, dummy_b = 2;

    Args args{};
    args.tensors[test_spec.lhs().args_slot] = {&dummy_a, makeShape(64, 32), makeStrides(32, 1)};
    args.tensors[test_spec.rhs().args_slot] = {&dummy_b, makeShape(32, 64), makeStrides(1, 32)};
    // output slot intentionally left null

    EXPECT_DEATH(validate(args, test_spec), "tensor \"C\" \\(slot 2\\) has null pointer");
}

TEST(ValidateDeathTest, AbortsOnMissingD0Tensor)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3;

    Args args{};
    args.tensors[test_spec_d0.lhs().args_slot] = {&dummy_a, makeShape(64, 32), makeStrides(32, 1)};
    args.tensors[test_spec_d0.rhs().args_slot] = {&dummy_b, makeShape(32, 64), makeStrides(1, 32)};
    args.tensors[test_spec_d0.output().args_slot] = {
        &dummy_c, makeShape(64, 64), makeStrides(64, 1)};
    // D0 (bias) slot intentionally left null

    EXPECT_DEATH(validate(args, test_spec_d0), "tensor \"bias\" \\(slot 3\\) has null pointer");
}

TEST(ValidateDeathTest, ReportsFirstMissingTensor)
{
    // All slots null — should report the first one (lhs = "A", slot 0)
    Args args{};

    EXPECT_DEATH(validate(args, test_spec), "tensor \"A\" \\(slot 0\\) has null pointer");
}
#endif
