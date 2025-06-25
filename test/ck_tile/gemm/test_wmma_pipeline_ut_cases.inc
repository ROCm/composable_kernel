// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

TYPED_TEST(TestCkTileWmmaPipeline, FitMultiWmma)
{
    std::vector<int> Ms{64, 128, 256, 512};
    constexpr int N = 256;
    constexpr int K = 128;

    for(int M : Ms)
        this->Run(M, N, K);
}
