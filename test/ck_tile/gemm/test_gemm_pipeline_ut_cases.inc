// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef TEST_GEMM_PIPELINE_UT_CASES_INC
#define TEST_GEMM_PIPELINE_UT_CASES_INC

TYPED_TEST(TEST_SUITE_NAME, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
    {
        if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                    ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
        }
        else
        {
            this->Run(M, N, K);
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N           = 1024;
    constexpr int K           = 320;
    constexpr int VecLoadSize = (std::is_same_v<typename TestFixture::ADataType, ck_tile::fp8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::bf8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::int8_t>)
                                    ? 16
                                    : 8;

    for(int M : Ms)
    {
        if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                    ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            if(M % VecLoadSize == 0)
            {
                this->Run(M, N, K);
            }
            else
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
        }
        else
        {
            this->Run(M, N, K);
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, PaddK)
{
    std::vector<int> Ms{128};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, NotSupportedArgument)
{
    constexpr int M = 512;
    constexpr int N = 1025;
    constexpr int K = 513;

    constexpr bool PadM = false;
    constexpr bool PadN = false;
    constexpr bool PadK = false;

    EXPECT_THROW((this->template Run<PadM, PadN, PadK>(M, N, K)), std::runtime_error);
}

#endif
