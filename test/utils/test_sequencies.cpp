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
    constexpr int DstVectorDim = 6;
    constexpr int DstScalarPerVector = 2;
    constexpr index_t nDim = 8;

    constexpr auto SliceLengths = Sequence<4, 8, 1, 1, 4, 1, 2, 1>{};

    // Generates a sequence (1, 1, 1, 1, 1, 1, 2, 1).
    // nDim gives the size of the sequence and DstVectorDim the position where to insert
    // value DstScalarPerVector. The rest of the sequence is filled with 1s.
    constexpr auto ScalarPerAccess = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

    constexpr auto Rem = SliceLengths % ScalarPerAccess;
    constexpr auto Zeros = Sequence<0, 0, 0, 0, 0, 0, 0, 0>{};

    constexpr bool is_valid = Rem == Zeros;

    EXPECT_TRUE(is_valid);
}
