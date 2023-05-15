#pragma once

TYPED_TEST(TestGemmSplitK_MK_KN, SmallM)
{
    std::vector<int> Ms{0, 1, 2, 3, 4, 5, 6};
    int N = 512;
    int K = 320;

    int StrideA = K;
    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_NK, SmallM)
{
    std::vector<int> Ms{0, 1, 2, 3, 4, 5, 6};
    int N = 512;
    int K = 320;

    int StrideA = K;
    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_KN, SmallM)
{
    std::vector<int> Ms{0, 1, 2, 3, 4, 5, 6};
    int N = 512;
    int K = 320;

    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_NK, SmallM)
{
    std::vector<int> Ms{0, 1, 2, 3, 4, 5, 6};
    int N = 512;
    int K = 320;

    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    int N = 512;
    int K = 320;

    int StrideA = K;
    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    int N = 512;
    int K = 320;

    int StrideA = K;
    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    int N = 512;
    int K = 320;

    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    int N = 512;
    int K = 320;

    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_KN, PaddK)
{
    std::vector<int> Ms{127};
    int N = 512;
    int K = 437;

    int StrideA = K;
    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_NK, PaddK)
{
    std::vector<int> Ms{127};
    int N = 512;
    int K = 437;

    int StrideA = K;
    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_KN, PaddK)
{
    std::vector<int> Ms{127};
    int N = 512;
    int K = 437;

    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_NK, PaddK)
{
    std::vector<int> Ms{127};
    int N = 512;
    int K = 437;

    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_KN, Regular)
{
    std::vector<int> Ms{512};
    int N = 512;
    int K = 512;

    int StrideA = K;
    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_MK_NK, Regular)
{
    std::vector<int> Ms{512};
    int N = 512;
    int K = 512;

    int StrideA = K;
    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_KN, Regular)
{
    std::vector<int> Ms{512};
    int N = 512;
    int K = 512;

    int StrideB = N;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmSplitK_KM_NK, Regular)
{
    std::vector<int> Ms{512};
    int N = 512;
    int K = 512;

    int StrideB = K;
    int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}
