// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Ensure that we have the required macros defined before proceeding
#ifndef TEST_SUITE_NAME
#error "TEST_SUITE_NAME must be defined before including this file"
#endif
#ifndef TEST_SUITE_PARAMS
#error "TEST_SUITE_PARAMS must be defined before including this file"
#endif

// Macros to help generate test names from the parameters given
// Concatenate is able to stitch the template parameters symbol together with the runtime args
// values
#define CONCATENATE_TEST_NAME(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS) \
    M##SIZE_M##_N##SIZE_N##_K##SIZE_K##_SKBlocks##NUM_SK_BLOCKS
// Helper macro to expand the arguments before passing them to CONCATENATE_TEST_NAME
#define MAKE_TEST_NAME(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS) \
    CONCATENATE_TEST_NAME(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS)

// Macro to add a test TEST_NAME to the TEST_SUITE_NAME with the given parameters
#define STREAM_K_TEST_INTERNAL(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS, TEST_NAME) \
    TYPED_TEST(TEST_SUITE_NAME, TEST_NAME)                                       \
    {                                                                            \
        ck_tile::index_t M     = SIZE_M;                                         \
        ck_tile::index_t N     = SIZE_N;                                         \
        ck_tile::index_t K     = SIZE_K;                                         \
        uint32_t num_sk_blocks = NUM_SK_BLOCKS;                                  \
                                                                                 \
        this->Run(M, N, K, num_sk_blocks);                                       \
    }

// Macro that generates a test name from the TEST_SUITE_TPARAMS symbol and the given parameters,
// then adds that test to test suite TEST_SUITE_NAME
#define STREAM_K_TEST(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS) \
    STREAM_K_TEST_INTERNAL(SIZE_M,                           \
                           SIZE_N,                           \
                           SIZE_K,                           \
                           NUM_SK_BLOCKS,                    \
                           MAKE_TEST_NAME(SIZE_M, SIZE_N, SIZE_K, NUM_SK_BLOCKS))

STREAM_K_TEST(1, 1, 1, 0)
STREAM_K_TEST(1, 1, 1, 1)

// TODO: fails for <= wave tile
// STREAM_K_TEST(16, 16, 16, 0)
// STREAM_K_TEST(16, 16, 16, 1)
// STREAM_K_TEST(32, 32, 16, 0)
// STREAM_K_TEST(32, 32, 16, 1)

STREAM_K_TEST(32, 32, 32, 0)
STREAM_K_TEST(32, 32, 32, 1)
STREAM_K_TEST(32, 32, 32, 2)
STREAM_K_TEST(32, 32, 32, 3)

/// Prime number odd offsets
STREAM_K_TEST(37, 32, 32, 0)
STREAM_K_TEST(37, 32, 32, 1)
STREAM_K_TEST(37, 32, 32, 2)
STREAM_K_TEST(37, 32, 32, 3)

STREAM_K_TEST(32, 37, 32, 0)
STREAM_K_TEST(32, 37, 32, 1)
STREAM_K_TEST(32, 37, 32, 2)
STREAM_K_TEST(32, 37, 32, 3)

// TODO: Fails
// STREAM_K_TEST(32, 32, 37, 0)
// STREAM_K_TEST(32, 32, 37, 1)
// STREAM_K_TEST(32, 32, 37, 2)
// STREAM_K_TEST(32, 32, 37, 3)

// TODO: Fails
STREAM_K_TEST(37, 32, 37, 0)
STREAM_K_TEST(37, 32, 37, 1)
STREAM_K_TEST(37, 32, 37, 2)
STREAM_K_TEST(37, 32, 37, 3)

STREAM_K_TEST(37, 37, 37, 0)
STREAM_K_TEST(37, 37, 37, 1)
STREAM_K_TEST(37, 37, 37, 2)
STREAM_K_TEST(37, 37, 37, 3)

/// Cubed sizes
STREAM_K_TEST(256, 256, 256, 0)
STREAM_K_TEST(256, 256, 256, 4)
STREAM_K_TEST(256, 256, 256, 8)

// TODO: Fails
// STREAM_K_TEST(272, 272, 272, 0)
// STREAM_K_TEST(272, 272, 272, 8)
// STREAM_K_TEST(272, 272, 272, 16)

STREAM_K_TEST(288, 288, 288, 0)
STREAM_K_TEST(288, 288, 288, 4)
STREAM_K_TEST(288, 288, 288, 8)

STREAM_K_TEST(512, 512, 512, 0)
STREAM_K_TEST(512, 512, 512, 8)
STREAM_K_TEST(512, 512, 512, 16)

// TODO: Fails
// STREAM_K_TEST(528, 528, 528, 0)
// STREAM_K_TEST(528, 528, 528, 8)
// STREAM_K_TEST(528, 528, 528, 16)

STREAM_K_TEST(544, 544, 544, 0)
STREAM_K_TEST(544, 544, 544, 8)
STREAM_K_TEST(544, 544, 544, 16)

/// Long M skinny N and K
STREAM_K_TEST(512, 1, 1, 0)
STREAM_K_TEST(512, 1, 1, 8)
STREAM_K_TEST(512, 1, 1, 16)

STREAM_K_TEST(512, 32, 32, 0)
STREAM_K_TEST(512, 32, 32, 8)
STREAM_K_TEST(512, 32, 32, 16)

/// Long M and N and skinny K
// TODO: Fails with core dump
// STREAM_K_TEST(512, 512, 1, 0)
// STREAM_K_TEST(512, 512, 1, 8)
// STREAM_K_TEST(512, 512, 1, 16)

STREAM_K_TEST(512, 512, 32, 0)
STREAM_K_TEST(512, 512, 32, 8)
STREAM_K_TEST(512, 512, 32, 16)

/// Long M and K and skinny N
STREAM_K_TEST(512, 1, 512, 0)
STREAM_K_TEST(512, 1, 512, 8)
STREAM_K_TEST(512, 1, 512, 16)

STREAM_K_TEST(512, 32, 512, 0)
STREAM_K_TEST(512, 32, 512, 8)
STREAM_K_TEST(512, 32, 512, 16)

/// Long K and skinny M and N
STREAM_K_TEST(1, 1, 512, 0)
STREAM_K_TEST(1, 1, 512, 8)
STREAM_K_TEST(1, 1, 512, 16)

STREAM_K_TEST(32, 32, 512, 0)
STREAM_K_TEST(32, 32, 512, 8)
STREAM_K_TEST(32, 32, 512, 16)

// TODO: Renable this test once reduction is implemented
TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_SKBlocks12)
{
    GTEST_SKIP() << "Skipping this test: There are precision issues with atomics due to >=3 WGs "
                    "contributing to each macro tile in C";

    ck_tile::index_t M     = 256;
    ck_tile::index_t N     = 256;
    ck_tile::index_t K     = 256;
    uint32_t num_sk_blocks = 12;

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
