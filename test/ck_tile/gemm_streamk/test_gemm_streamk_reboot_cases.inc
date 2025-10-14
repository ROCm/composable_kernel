// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_M256_N256_K256_EdgeCase)
{
    ck_tile::index_t M = 256;
    ck_tile::index_t N = 256;
    ck_tile::index_t K = 256;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M10240_N6656_K256_DPOnly)
{

    ck_tile::index_t M = 10240; // This is 4 * 256
    ck_tile::index_t N = 6656;  // This is 26 * 256
    ck_tile::index_t K = 256;

    // Since 4 * 256 = 104, this means that the number of tiles will evenly divide the grid on
    // gfx90a to test DP only.

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M512_N512_K2048_SKOnly)
{

    ck_tile::index_t M = 512;
    ck_tile::index_t N = 512;
    ck_tile::index_t K = 2048;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_M768_N17920_K2048_DP2TSK)
{
    ck_tile::index_t M = 768;   // This is 3 * 256
    ck_tile::index_t N = 17920; // This is 70 * 256
    ck_tile::index_t K = 2048;

    // We have a total of 3 * 70 = 210 tiles in C.
    // On gfx90a, this will give us 104 tiles for DP and 106 tiles for 2 Tile SK.
    // Assuming 104 CUs and occupancy of 1.

    this->Run(M, N, K);
}
