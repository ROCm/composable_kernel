#pragma once

TYPED_TEST(TestCkTileGemmMemPipeline, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipeline, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipeline, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMemPipeline, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}
