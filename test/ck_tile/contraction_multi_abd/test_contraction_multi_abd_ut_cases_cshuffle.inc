// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// G=1, M=1, N=1, K=1 (simplest case: single-dim contraction)
TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G1_M1_N1_K1_256x256x256)
{
    EXPECT_TRUE((this->template Run<1, 1, 1, 1>({2}, {256}, {256}, {256})));
}

// G=1, M=2, N=2, K=2 (multi-dim contraction matching old CK example 61)
TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G1_M2_N2_K2_small)
{
    EXPECT_TRUE((this->template Run<1, 2, 2, 2>({2}, {4, 64}, {8, 32}, {4, 16})));
}

// G=2, M=1, N=1, K=1 (multi-batch with single M/N/K)
TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G2_M1_N1_K1_256x256x256)
{
    EXPECT_TRUE((this->template Run<2, 1, 1, 1>({1, 2}, {256}, {256}, {256})));
}

TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G2_M1_N1_K1_padded_g_strides)
{
    EXPECT_TRUE((this->template Run<2, 1, 1, 1>({2, 2}, {128}, {128}, {64}, true)));
}

// G=1, M=1, N=1, K=1 with larger problem
TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G1_M1_N1_K1_512x512x256)
{
    EXPECT_TRUE((this->template Run<1, 1, 1, 1>({4}, {512}, {512}, {256})));
}

TYPED_TEST(TestCkTileContractionMultiABD, CShuffle_G1_M1_N1_K1_padded_edge)
{
    EXPECT_TRUE((this->template Run<1, 1, 1, 1, false, true, true>({1}, {128}, {136}, {72})));
}
