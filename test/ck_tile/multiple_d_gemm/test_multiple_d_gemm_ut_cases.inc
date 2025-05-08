#pragma once

TYPED_TEST(TestCkTileMultipleDGemm, Basic)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 512;
    this->Run(M, N, K);
}
