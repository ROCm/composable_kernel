#pragma once

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_256x512x256)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_512x256x256)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_512x512x256)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_256x256x256)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_512x768x256)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_512x1280x256)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_256x1280x256)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_768x512x256)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_1280x512x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_1280x256x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x512x256)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x256x256)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x512x256)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x256x256)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x768x256)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_512x1280x256)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_256x1280x256)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_768x512x256)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x512x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch1_1280x256x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_256x256x512)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_512x768x512)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_512x1280x512)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_256x1280x512)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_768x512x512)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_1280x512x512)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch2_1280x256x512)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_256x512x512)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_512x256x512)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_512x512x512)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_256x256x512)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_512x768x512)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_512x1280x512)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_256x1280x512)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_768x512x512)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_1280x512x512)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDMultiplyMultiplyKBatch2_1280x256x512)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;
    this->Run(M, N, K, kBatch);
}
