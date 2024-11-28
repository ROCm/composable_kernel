// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

//------------------------------------------------------------------------------------------------
//              INTERWAVE SCHEDULER
//------------------------------------------------------------------------------------------------

TYPED_TEST(TestCkTileGemmMemPipelineInterwave, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineInterwave, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineInterwave, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineInterwave, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}

//------------------------------------------------------------------------------------------------
//              INTRAWAVE SCHEDULER
//------------------------------------------------------------------------------------------------

TYPED_TEST(TestCkTileGemmMemPipelineIntrawave, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineIntrawave, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineIntrawave, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipelineIntrawave, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}
