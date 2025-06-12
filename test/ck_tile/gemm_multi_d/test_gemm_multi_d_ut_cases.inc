#pragma once

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_256x512x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_512x256x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_512x512x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_256x256x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_512x768x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_512x1280x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_256x1280x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_768x512x256, Basic)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_1280x512x256, Basic)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDAddKBatch1_1280x256x256, Basic)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x512x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x256x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x512x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x256x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x768x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x1280x256, Basic)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x1280x256, Basic)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_768x512x256, Basic)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x512x256, Basic)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x256x256, Basic)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}
