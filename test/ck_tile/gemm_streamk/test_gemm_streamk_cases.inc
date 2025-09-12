// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_DP)
{

    ck_tile::index_t M     = 256;
    ck_tile::index_t N     = 256;
    ck_tile::index_t K     = 256;
    uint32_t num_sk_blocks = 0;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_SKBlocks4)
{

    ck_tile::index_t M     = 256;
    ck_tile::index_t N     = 256;
    ck_tile::index_t K     = 256;
    uint32_t num_sk_blocks = 4;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_SKBlocks12)
{

    ck_tile::index_t M     = 256;
    ck_tile::index_t N     = 256;
    ck_tile::index_t K     = 256;
    uint32_t num_sk_blocks = 12;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_SKBlocks8)
{

    ck_tile::index_t M     = 256;
    ck_tile::index_t N     = 256;
    ck_tile::index_t K     = 256;
    uint32_t num_sk_blocks = 8;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M512_N512_K512_DP)
{

    ck_tile::index_t M     = 512;
    ck_tile::index_t N     = 512;
    ck_tile::index_t K     = 512;
    uint32_t num_sk_blocks = 0;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M512_N512_K512_SKBlocks16)
{

    ck_tile::index_t M     = 512;
    ck_tile::index_t N     = 512;
    ck_tile::index_t K     = 512;
    uint32_t num_sk_blocks = 16;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M512_N512_K512_SKBlocks8)
{

    ck_tile::index_t M     = 512;
    ck_tile::index_t N     = 512;
    ck_tile::index_t K     = 512;
    uint32_t num_sk_blocks = 8;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M3840_N4096_K4096_DP)
{

    ck_tile::index_t M     = 3840;
    ck_tile::index_t N     = 4096;
    ck_tile::index_t K     = 4096;
    uint32_t num_sk_blocks = 0;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M3840_N4096_K4096_SKBlocks64)
{

    ck_tile::index_t M     = 3840;
    ck_tile::index_t N     = 4096;
    ck_tile::index_t K     = 4096;
    uint32_t num_sk_blocks = 64;

    this->Run(M, N, K, num_sk_blocks);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Unsupported_Reduction)
{

    ck_tile::index_t M     = 3840;
    ck_tile::index_t N     = 4096;
    ck_tile::index_t K     = 4096;
    uint32_t num_sk_blocks = 64;

    EXPECT_THROW(this->Run(M, N, K, num_sk_blocks, ck_tile::StreamKReductionStrategy::Reduction),
                 std::runtime_error);
}
