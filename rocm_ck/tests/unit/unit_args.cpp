// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/args.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using ::rocm_ck::Args;
using ::rocm_ck::kMaxRank;
using ::rocm_ck::kMaxScalars;
using ::rocm_ck::kMaxTensors;
using ::rocm_ck::makeShape;
using ::rocm_ck::makeStrides;
using ::rocm_ck::ScalarValue;
using ::rocm_ck::TensorArg;
using ::testing::ElementsAre;

namespace {

// ============================================================================
// TensorArg ABI
// ============================================================================

TEST(TensorArg, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<TensorArg>); }

TEST(TensorArg, HasStandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<TensorArg>); }

TEST(TensorArg, Occupies80Bytes)
{
    // ptr(8) + lengths(6*4=24) + strides(6*8=48) = 80
    EXPECT_EQ(sizeof(TensorArg), 80);
}

TEST(TensorArg, AlignsTo8Bytes) { EXPECT_EQ(alignof(TensorArg), 8); }

TEST(TensorArg, PlacesFieldsAtExpectedOffsets)
{
    EXPECT_EQ(offsetof(TensorArg, ptr), 0);
    EXPECT_EQ(offsetof(TensorArg, lengths), 8);
    EXPECT_EQ(offsetof(TensorArg, strides), 32);
}

// ============================================================================
// ScalarValue ABI
// ============================================================================

TEST(ScalarValue, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<ScalarValue>); }

TEST(ScalarValue, Occupies8Bytes)
{
    // Union of float(4), int32(4), uint32(4), double(8) -> 8 bytes
    EXPECT_EQ(sizeof(ScalarValue), 8);
}

// ============================================================================
// Args ABI
// ============================================================================

TEST(Args, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<Args>); }

TEST(Args, HasStandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<Args>); }

TEST(Args, Occupies1552Bytes)
{
    // 16 tensors * 80 + 16 scalars * 8 + batch_count(4) + pad(4)
    // + 16 batch_strides * 8 + workspace_ptr(8) = 1280 + 128 + 8 + 128 + 8 = 1552
    EXPECT_EQ(sizeof(Args), 1552);
}

TEST(Args, AlignsTo8Bytes) { EXPECT_EQ(alignof(Args), 8); }

TEST(Args, FitsWithin4KBKernargBudget)
{
    // HSA minimum kernarg size is 4096 bytes
    EXPECT_LE(sizeof(Args), 4096);
}

// ============================================================================
// Capacity constants
// ============================================================================

TEST(Args, DefinesExpectedCapacityLimits)
{
    EXPECT_EQ(kMaxRank, 6);
    EXPECT_EQ(kMaxTensors, 16);
    EXPECT_EQ(kMaxScalars, 16);
}

// ============================================================================
// ScalarValue union access
// ============================================================================

TEST(ScalarValue, StoresAndRetrievesFloat)
{
    ScalarValue sv{};
    sv.f32 = 3.14f;
    EXPECT_FLOAT_EQ(sv.f32, 3.14f);
}

TEST(ScalarValue, StoresAndRetrievesInt32)
{
    ScalarValue sv{};
    sv.i32 = -42;
    EXPECT_EQ(sv.i32, -42);
}

TEST(ScalarValue, StoresAndRetrievesDouble)
{
    ScalarValue sv{};
    sv.f64 = 2.718281828;
    EXPECT_DOUBLE_EQ(sv.f64, 2.718281828);
}

TEST(ScalarValue, StoresAndRetrievesUInt32)
{
    ScalarValue sv{};
    sv.u32 = 0xDEADBEEF;
    EXPECT_EQ(sv.u32, 0xDEADBEEF);
}

// ============================================================================
// Args field coverage — batch_strides and workspace_ptr
// ============================================================================

TEST(Args, BatchStridesFieldExists)
{
    Args args{};
    args.batch_strides[0]               = 12345;
    args.batch_strides[kMaxTensors - 1] = -99;
    EXPECT_EQ(args.batch_strides[0], 12345);
    EXPECT_EQ(args.batch_strides[kMaxTensors - 1], -99);
}

TEST(Args, WorkspacePtrFieldExists)
{
    Args args{};
    int dummy          = 42;
    args.workspace_ptr = &dummy;
    EXPECT_EQ(args.workspace_ptr, &dummy);
}

TEST(Args, BatchCountFieldExists)
{
    Args args{};
    args.batch_count = 8;
    EXPECT_EQ(args.batch_count, 8);
}

// ============================================================================
// Boundary access tests
// ============================================================================

TEST(Args, BoundaryAccessToTensors)
{
    Args args{};
    // Access last tensor slot (kMaxTensors - 1 = 15)
    args.tensors[kMaxTensors - 1].ptr = nullptr;
    EXPECT_EQ(args.tensors[kMaxTensors - 1].ptr, nullptr);
}

TEST(Args, BoundaryAccessToScalars)
{
    Args args{};
    // Access last scalar slot (kMaxScalars - 1 = 15)
    args.scalars[kMaxScalars - 1].f32 = 1.0f;
    EXPECT_FLOAT_EQ(args.scalars[kMaxScalars - 1].f32, 1.0f);
}

TEST(TensorArg, BoundaryAccessToLengthsAndStrides)
{
    TensorArg ta{};
    // Access last rank dimension (kMaxRank - 1 = 5)
    ta.lengths[kMaxRank - 1] = 42;
    ta.strides[kMaxRank - 1] = 99;
    EXPECT_EQ(ta.lengths[kMaxRank - 1], 42);
    EXPECT_EQ(ta.strides[kMaxRank - 1], 99);
}

// ============================================================================
// makeShape
// ============================================================================

TEST(MakeShape, ZeroFillsUnusedDimensions)
{
    EXPECT_THAT(makeShape(128, 64), ElementsAre(128, 64, 0, 0, 0, 0));
}

TEST(MakeShape, FillsAllSixDimensions)
{
    EXPECT_THAT(makeShape(2, 3, 4, 5, 6, 7), ElementsAre(2, 3, 4, 5, 6, 7));
}

TEST(MakeShape, SingleDimension) { EXPECT_THAT(makeShape(1024), ElementsAre(1024, 0, 0, 0, 0, 0)); }

// ============================================================================
// makeStrides
// ============================================================================

TEST(MakeStrides, ZeroFillsUnusedDimensions)
{
    EXPECT_THAT(makeStrides(256, 1), ElementsAre(256, 1, 0, 0, 0, 0));
}

TEST(MakeStrides, HandlesLargeInt64Values)
{
    constexpr int64_t large = 1LL << 40;
    EXPECT_THAT(makeStrides(large, 1), ElementsAre(large, 1, 0, 0, 0, 0));
}

} // namespace
