#pragma once

TYPED_TEST(TestCkTileBatchedGemm, Basic)
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 512;
    this->Run(M, N, K);
}
