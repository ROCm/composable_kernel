// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

using namespace ck;

// =============================================================================
// Tests for convert_visible_to_hidden_id functor
// =============================================================================

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor3D)
{
    // For a 3D packed descriptor created via make_naive_tensor_descriptor_packed,
    // the visible dimension IDs are Sequence<1, 2, 3> (hidden dim 0 is the element space)
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    // Verify the visible dimension IDs for a packed descriptor
    constexpr auto visible_ids = DescType::GetVisibleDimensionIds();
    using ExpectedVisibleIds   = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<decltype(visible_ids), const ExpectedVisibleIds>::value));

    // Test the functor
    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // Visible ID 0 -> Hidden ID 1
    constexpr auto hidden_0 = functor(Number<0>{});
    EXPECT_EQ(hidden_0, 1);

    // Visible ID 1 -> Hidden ID 2
    constexpr auto hidden_1 = functor(Number<1>{});
    EXPECT_EQ(hidden_1, 2);

    // Visible ID 2 -> Hidden ID 3
    constexpr auto hidden_2 = functor(Number<2>{});
    EXPECT_EQ(hidden_2, 3);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor2D)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<32>{}, Number<64>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 2D packed: visible IDs are Sequence<1, 2>
    constexpr auto hidden_0 = functor(Number<0>{});
    constexpr auto hidden_1 = functor(Number<1>{});

    EXPECT_EQ(hidden_0, 1);
    EXPECT_EQ(hidden_1, 2);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor1D)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(make_tuple(Number<128>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 1D packed: visible IDs are Sequence<1>
    constexpr auto hidden_0 = functor(Number<0>{});
    EXPECT_EQ(hidden_0, 1);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor4D)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<2>{}, Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 4D packed: visible IDs are Sequence<1, 2, 3, 4>
    EXPECT_EQ(functor(Number<0>{}), 1);
    EXPECT_EQ(functor(Number<1>{}), 2);
    EXPECT_EQ(functor(Number<2>{}), 3);
    EXPECT_EQ(functor(Number<3>{}), 4);
}

// =============================================================================
// Tests for convert_visible_ids_to_hidden_ids functor
// =============================================================================

TEST(ConvertVisibleIdsToHiddenIds, SingleElement)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert single visible ID
    constexpr auto result = functor(Sequence<0>{});
    using ExpectedResult  = Sequence<1>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, MultipleElements)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert multiple visible IDs
    constexpr auto result = functor(Sequence<0, 2>{});
    using ExpectedResult  = Sequence<1, 3>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, AllElements)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert all visible IDs
    constexpr auto result = functor(Sequence<0, 1, 2>{});
    using ExpectedResult  = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, ReversedOrder)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert visible IDs in reverse order
    constexpr auto result = functor(Sequence<2, 1, 0>{});
    using ExpectedResult  = Sequence<3, 2, 1>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, EmptySequence)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert empty sequence
    constexpr auto result = functor(Sequence<>{});
    using ExpectedResult  = Sequence<>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, HighDimensional)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<2>{}, Number<4>{}, Number<8>{}, Number<16>{}, Number<32>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert subset of visible IDs
    constexpr auto result = functor(Sequence<1, 3>{});
    using ExpectedResult  = Sequence<2, 4>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

// =============================================================================
// Tests for generate_arithmetic_sequence_from_scan functor
// =============================================================================

TEST(GenerateArithmeticSequenceFromScan, SingleRange)
{
    // Scan sequence: <0, 3> means:
    //   - Index 0: range from (base + 0) to (base + 3) = Sequence<base, base+1, base+2>
    using ScanSeq          = Sequence<0, 3>;
    constexpr index_t base = 5;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result = functor(Number<0>{});
    using ExpectedResult  = Sequence<5, 6, 7>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(GenerateArithmeticSequenceFromScan, MultipleRanges)
{
    // Scan sequence: <0, 2, 5> means:
    //   - Index 0: range from (base + 0) to (base + 2) = 2 elements
    //   - Index 1: range from (base + 2) to (base + 5) = 3 elements
    using ScanSeq          = Sequence<0, 2, 5>;
    constexpr index_t base = 3;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // First range: base + [0, 2) = Sequence<3, 4>
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<3, 4>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    // Second range: base + [2, 5) = Sequence<5, 6, 7>
    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<5, 6, 7>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));
}

TEST(GenerateArithmeticSequenceFromScan, SingleElementRanges)
{
    // Scan sequence with single-element ranges: <0, 1, 2, 3>
    using ScanSeq          = Sequence<0, 1, 2, 3>;
    constexpr index_t base = 10;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // Each range contains exactly one element
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<10>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<11>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));

    constexpr auto result_2 = functor(Number<2>{});
    using ExpectedResult2   = Sequence<12>;
    EXPECT_TRUE((is_same<decltype(result_2), const ExpectedResult2>::value));
}

TEST(GenerateArithmeticSequenceFromScan, ZeroBase)
{
    // Test with base = 0
    using ScanSeq          = Sequence<0, 2, 4>;
    constexpr index_t base = 0;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<0, 1>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<2, 3>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));
}

TEST(GenerateArithmeticSequenceFromScan, VariableSizeRanges)
{
    // Scan sequence with variable-size ranges: <0, 1, 4, 6, 10>
    using ScanSeq          = Sequence<0, 1, 4, 6, 10>;
    constexpr index_t base = 2;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // Range 0: [2+0, 2+1) = Sequence<2>
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<2>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    // Range 1: [2+1, 2+4) = Sequence<3, 4, 5>
    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<3, 4, 5>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));

    // Range 2: [2+4, 2+6) = Sequence<6, 7>
    constexpr auto result_2 = functor(Number<2>{});
    using ExpectedResult2   = Sequence<6, 7>;
    EXPECT_TRUE((is_same<decltype(result_2), const ExpectedResult2>::value));

    // Range 3: [2+6, 2+10) = Sequence<8, 9, 10, 11>
    constexpr auto result_3 = functor(Number<3>{});
    using ExpectedResult3   = Sequence<8, 9, 10, 11>;
    EXPECT_TRUE((is_same<decltype(result_3), const ExpectedResult3>::value));
}

TEST(GenerateArithmeticSequenceFromScan, LargeBase)
{
    // Test with a larger base value
    using ScanSeq          = Sequence<0, 3>;
    constexpr index_t base = 100;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result = functor(Number<0>{});
    using ExpectedResult  = Sequence<100, 101, 102>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

// =============================================================================
// Integration tests - verify functors work together in transform_tensor_descriptor context
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, TransformPreservesMapping)
{
    // Create a simple packed 2D descriptor
    constexpr auto desc = make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}));

    using DescType = decltype(desc);

    // Verify the descriptor structure
    EXPECT_EQ(DescType::GetNumOfVisibleDimension(), 2);
    EXPECT_EQ(DescType::GetNumOfHiddenDimension(), 3); // 1 (element space) + 2 (visible dims)

    // Test that convert_visible_to_hidden_id preserves the expected mapping
    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // The hidden IDs should be consecutive starting from 1
    constexpr auto hidden_0 = functor(Number<0>{});
    constexpr auto hidden_1 = functor(Number<1>{});

    EXPECT_EQ(hidden_0, 1);
    EXPECT_EQ(hidden_1, 2);
    EXPECT_EQ(hidden_1 - hidden_0, 1); // Should be consecutive
}

TEST(TensorDescriptorFunctorsIntegration, ConvertIdsMatchDirectAccess)
{
    // Verify that the functor produces the same result as direct access
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<2>{}, Number<4>{}, Number<8>{}));

    using DescType = decltype(desc);

    constexpr auto visible_ids = DescType::GetVisibleDimensionIds();
    constexpr auto functor     = convert_visible_to_hidden_id<DescType>{};

    // Functor result should match direct sequence access
    EXPECT_EQ(functor(Number<0>{}), visible_ids.At(Number<0>{}));
    EXPECT_EQ(functor(Number<1>{}), visible_ids.At(Number<1>{}));
    EXPECT_EQ(functor(Number<2>{}), visible_ids.At(Number<2>{}));
}

TEST(TensorDescriptorFunctorsIntegration, BatchConvertMatchesSingle)
{
    // Verify that batch conversion produces the same result as individual conversions
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto single_functor = convert_visible_to_hidden_id<DescType>{};
    constexpr auto batch_functor  = convert_visible_ids_to_hidden_ids<DescType>{};

    constexpr auto batch_result = batch_functor(Sequence<0, 1, 2>{});

    // Each element should match the single conversion
    EXPECT_EQ(batch_result.At(Number<0>{}), single_functor(Number<0>{}));
    EXPECT_EQ(batch_result.At(Number<1>{}), single_functor(Number<1>{}));
    EXPECT_EQ(batch_result.At(Number<2>{}), single_functor(Number<2>{}));
}
