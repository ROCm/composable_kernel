// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestCkTileContractionMultiABD, Default2D_G1_M1_N1_K1_256x256x256)
{
    EXPECT_TRUE((this->template Run<1, 1, 1, 1>({2}, {256}, {256}, {256})));
}

TYPED_TEST(TestCkTileContractionMultiABD, Default2D_G1_M2_N2_K2_small)
{
    EXPECT_TRUE((this->template Run<1, 2, 2, 2>({2}, {4, 64}, {8, 32}, {4, 16})));
}

TYPED_TEST(TestCkTileContractionMultiABD, Default2D_G2_M1_N1_K1_256x256x256)
{
    EXPECT_TRUE((this->template Run<2, 1, 1, 1>({1, 2}, {256}, {256}, {256})));
}

TYPED_TEST(TestCkTileContractionMultiABD, Default2D_G2_M1_N1_K1_padded_g_strides)
{
    EXPECT_TRUE((this->template Run<2, 1, 1, 1>({2, 2}, {128}, {128}, {64}, true)));
}

TYPED_TEST(TestCkTileContractionMultiABD, Default2D_G1_M1_N1_K1_padded_edge)
{
    EXPECT_TRUE((this->template Run<1, 1, 1, 1, false, true, true>({1}, {128}, {136}, {72})));
}
