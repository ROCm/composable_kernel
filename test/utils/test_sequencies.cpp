// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "ck/utility/common_header.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/number.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

using namespace ck;

TEST(Sequence, TestCreatingAccessSequence)
{
    constexpr int DstVectorDim       = 6;
    constexpr int DstScalarPerVector = 2;
    constexpr index_t nDim           = 8;

    constexpr auto SliceLengths = Sequence<4, 8, 1, 1, 4, 1, 2, 1>{};

    // Generates a sequence (1, 1, 1, 1, 1, 1, 2, 1).
    // nDim gives the size of the sequence and DstVectorDim the position where to insert
    // value DstScalarPerVector. The rest of the sequence is filled with 1s.
    constexpr auto ScalarPerAccess = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

    constexpr auto Rem   = SliceLengths % ScalarPerAccess;
    constexpr auto Zeros = Sequence<0, 0, 0, 0, 0, 0, 0, 0>{};

    constexpr bool is_valid = Rem == Zeros;

    EXPECT_TRUE(is_valid);
}

TEST(Sequence, TestCreatingSequenceFromTuple)
{
    constexpr ck::index_t I0 = 0;
    constexpr ck::index_t I1 = 1;

    constexpr auto seq = to_sequence(make_tuple(Number<I0>{}, Number<I1>{}));

    EXPECT_EQ(seq[0], I0);
    EXPECT_EQ(seq[1], I1);
}

TEST(Sequence, TestCreatingSequenceFromStaticFor)
{
    constexpr ck::index_t num_pairs = 2;
    constexpr auto I1               = Number<1>{};
    constexpr auto I2               = Number<2>{};
    static_for<0, num_pairs, 1>{}([&](auto i_pair) {
        constexpr auto idx_1d_0 = I2 * i_pair;
        constexpr auto idx_1d_1 = I2 * i_pair + I1;

        constexpr auto seq = to_sequence(make_tuple(idx_1d_0, idx_1d_1));

        if constexpr(i_pair == 0)
        {
            EXPECT_EQ(seq[0], 0);
            EXPECT_EQ(seq[1], 1);
        }
        else if constexpr(i_pair == 1)
        {
            EXPECT_EQ(seq[0], 2);
            EXPECT_EQ(seq[1], 3);
        }
    });
}

TEST(Sequence, TestSumOfElements)
{
    constexpr auto seq = Sequence<1, 2, 3, 4>{};
    index_t sum        = reduce_on_sequence(seq, math::plus<index_t>{}, Number<0>{});
    EXPECT_EQ(sum, 10);
}

TEST(Sequence, TestMultiplyOfElements)
{
    constexpr auto seq = Sequence<1, 2, 3, 4>{};
    index_t sum        = reduce_on_sequence(seq, math::multiplies{}, Number<1>{});
    EXPECT_EQ(sum, 24);
}
