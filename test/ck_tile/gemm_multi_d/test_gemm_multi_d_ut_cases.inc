#pragma once

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1, Basic)
{
    constexpr int M      = 1024;
    constexpr int N      = 512;
    constexpr int K      = 1024;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch2, Basic)
{
    constexpr int M      = 1024;
    constexpr int N      = 512;
    constexpr int K      = 1024;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyKBatch1, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyKBatch2, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}
