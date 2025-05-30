#pragma once

TYPED_TEST(TestCkTileGemmMultiDAdd, Basic)
{
    constexpr int M = 1024;
    constexpr int N = 512;
    constexpr int K = 1024;
    this->Run(M, N, K);
}

TYPED_TEST(TestCkTileGemmMultiDMultiply, Basic)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 256;
    this->Run(M, N, K);
}
