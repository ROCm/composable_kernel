#pragma once

TYPED_TEST(TestCkTileMultipleDGemm, Basic)
{
    constexpr int M = 32;
    constexpr int N = 256;
    constexpr int K = 256;
    this->Run(M, N, K);
}
