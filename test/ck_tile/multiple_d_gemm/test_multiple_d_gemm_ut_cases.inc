#pragma once

TYPED_TEST(TestCkTileMultipleDGemm, Basic)
{
    constexpr int M = 3840;
    constexpr int N = 4096;
    constexpr int K = 4096;
    this->Run(M, N, K);
}
