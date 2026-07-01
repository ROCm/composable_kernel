// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, Regular)
{
    std::vector<int> Ms{128, 256};
    std::vector<int> Ns{128, 256};
    std::vector<int> Ks;
    // K must be multiple of ScaleBlockSize (16 or 32) and K_Tile
    for(auto K_count : {1, 2, 3, 4})
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
