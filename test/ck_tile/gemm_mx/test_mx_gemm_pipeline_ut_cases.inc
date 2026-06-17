// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"

TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    // K must be multiple of ScaleBlockSize (16 or 32), and >= K_Tile
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, SmallM)
{
    std::vector<int> Ms{1, 2, 4, 8, 16};
    constexpr int N = 64;
    std::vector<int> Ks;
    // K must be multiple of ScaleBlockSize (16 or 32) and K_Tile
    for(auto K_count : {2, 3, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            this->Run(M, N, K);
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM)
{
    std::vector<int> Ms{32, 64, 128, 256};
    std::vector<int> Ns{96, 128}; // 96 tests non-tile-aligned N
    std::vector<int> Ks;
    for(auto K_count : {2, 3, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int N : Ns)
        {
            for(int K : Ks)
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, LargeSize) { this->Run(512, 512, 512); }
