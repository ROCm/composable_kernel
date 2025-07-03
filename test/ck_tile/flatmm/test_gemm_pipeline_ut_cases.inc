// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef TEST_GEMM_PIPELINE_UT_CASES_INC
#define TEST_GEMM_PIPELINE_UT_CASES_INC

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle)
{
    constexpr int M           = 2048;
    constexpr int N           = 4096;
    constexpr int K           = 5120;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

#endif
