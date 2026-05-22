// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/signature.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::AddOp;
using ::rocm_ck::DataType;
using ::rocm_ck::FastGeluOp;
using ::rocm_ck::GemmOp;
using ::rocm_ck::kMaxOps;
using ::rocm_ck::kMaxScalars;
using ::rocm_ck::kMaxTensors;
using ::rocm_ck::Layout;
using ::rocm_ck::MulOp;
using ::rocm_ck::Op;
using ::rocm_ck::Quantization;
using ::rocm_ck::ReluOp;
using ::rocm_ck::Scalar;
using ::rocm_ck::SigmoidOp;
using ::rocm_ck::Signature;
using ::rocm_ck::Tensor;

// ============================================================================
// Signature construction
// ============================================================================

TEST(Signature, DefaultsToNoDtype)
{
    constexpr Signature sig{};
    EXPECT_FALSE(sig.dtype.has_value());
}

TEST(Signature, StoresExplicitDtype)
{
    constexpr Signature sig{.dtype = DataType::FP16};
    EXPECT_TRUE(sig.dtype.has_value());
    EXPECT_EQ(*sig.dtype, DataType::FP16);
}

// ============================================================================
// Tensor
// ============================================================================

TEST(Tensor, DefaultsToAutoLayoutAndRankZero)
{
    constexpr Tensor t{.name = "A"};
    EXPECT_EQ(t.name, "A");
    EXPECT_FALSE(t.dtype.has_value());
    EXPECT_EQ(t.rank, 0);
    EXPECT_EQ(t.layout, Layout::Auto);
}

TEST(Tensor, StoresAllExplicitFields)
{
    constexpr Tensor t{.name = "Q", .dtype = DataType::FP32, .rank = 3, .layout = Layout::Row};
    EXPECT_EQ(t.name, "Q");
    EXPECT_EQ(*t.dtype, DataType::FP32);
    EXPECT_EQ(t.rank, 3);
    EXPECT_EQ(t.layout, Layout::Row);
}

// ============================================================================
// Scalar
// ============================================================================

TEST(Scalar, DefaultsToFP32Dtype)
{
    constexpr Scalar s{.name = "alpha"};
    EXPECT_EQ(s.name, "alpha");
    EXPECT_EQ(s.dtype, DataType::FP32);
}

TEST(Scalar, StoresExplicitDtype)
{
    constexpr Scalar s{.name = "scale", .dtype = DataType::FP16};
    EXPECT_EQ(s.dtype, DataType::FP16);
}

// ============================================================================
// Op variant
// ============================================================================

TEST(Op, DefaultsToMonostate)
{
    constexpr Op op{};
    EXPECT_TRUE(std::holds_alternative<std::monostate>(op));
}

TEST(Op, HoldsGemmOp)
{
    constexpr Op op = GemmOp{.lhs = "A", .rhs = "B", .out = "C"};
    EXPECT_TRUE(std::holds_alternative<GemmOp>(op));
}

TEST(Op, HoldsAllUnaryOpTypes)
{
    constexpr Op relu = ReluOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<ReluOp>(relu));

    constexpr Op gelu = FastGeluOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<FastGeluOp>(gelu));

    constexpr Op sigmoid = SigmoidOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<SigmoidOp>(sigmoid));
}

TEST(Op, HoldsAllBinaryOpTypes)
{
    constexpr Op add = AddOp{.lhs = "X", .rhs = "Y", .out = "Z"};
    EXPECT_TRUE(std::holds_alternative<AddOp>(add));

    constexpr Op mul = MulOp{.lhs = "X", .rhs = "Y", .out = "Z"};
    EXPECT_TRUE(std::holds_alternative<MulOp>(mul));
}

// ============================================================================
// GemmOp defaults
// ============================================================================

TEST(GemmOp, DefaultsAccDtypeToFP32)
{
    constexpr GemmOp gemm{.lhs = "A", .rhs = "B", .out = "C"};
    EXPECT_EQ(gemm.acc_dtype, DataType::FP32);
}

// ============================================================================
// Quantization
// ============================================================================

TEST(Quantization, DefaultsToFP32ScaleAndGroupSize128)
{
    constexpr Quantization q{.scale_name = "scale"};
    EXPECT_EQ(q.scale_name, "scale");
    EXPECT_EQ(q.scale_dtype, DataType::FP32);
    EXPECT_EQ(q.group_size, 128);
}

TEST(Quantization, StoresExplicitFields)
{
    constexpr Quantization q{.scale_name = "bq", .scale_dtype = DataType::FP16, .group_size = 64};
    EXPECT_EQ(q.scale_name, "bq");
    EXPECT_EQ(q.scale_dtype, DataType::FP16);
    EXPECT_EQ(q.group_size, 64);
}

TEST(Tensor, DefaultsToNoQuantize)
{
    constexpr Tensor t{.name = "B"};
    EXPECT_FALSE(t.quantize.has_value());
}

TEST(Tensor, StoresQuantizeMetadata)
{
    constexpr Tensor t{
        .name  = "B",
        .dtype = DataType::I4,
        .quantize =
            Quantization{.scale_name = "scale", .scale_dtype = DataType::FP32, .group_size = 128}};
    EXPECT_TRUE(t.quantize.has_value());
    EXPECT_EQ(t.quantize->scale_name, "scale");
    EXPECT_EQ(t.quantize->scale_dtype, DataType::FP32);
    EXPECT_EQ(t.quantize->group_size, 128);
}

// ============================================================================
// Capacity constants
// ============================================================================

TEST(Signature, DefinesExpectedCapacityLimits)
{
    EXPECT_EQ(kMaxTensors, 16);
    EXPECT_EQ(kMaxScalars, 16);
    EXPECT_EQ(kMaxOps, 8);
}
