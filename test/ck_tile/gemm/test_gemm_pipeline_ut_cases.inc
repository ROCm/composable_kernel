// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

TYPED_TEST(TestCkTileGemmPipeline, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmPipeline, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmPipeline, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmPipeline, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmPipeline, NotSupportedArgument)
{
    constexpr int M = 512;
    constexpr int N = 1025;
    constexpr int K = 513;

    constexpr bool PadM = false;
    constexpr bool PadN = false;
    constexpr bool PadK = false;

    EXPECT_THROW((this->template Run<PadM, PadN, PadK>(M, N, K)), std::runtime_error);
}
