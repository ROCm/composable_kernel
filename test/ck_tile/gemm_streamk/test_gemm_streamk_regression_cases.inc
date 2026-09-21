// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_Remainder_Along_K)
{

    ck_tile::index_t M = 64;
    ck_tile::index_t N = 270128;
    ck_tile::index_t K = 112;

    this->Run(M, N, K);
}
