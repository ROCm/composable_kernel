// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"

TYPED_TEST(TEST_SUITE_NAME, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 3, 4, 10, 11})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ADataType, ck_tile::fp16_t> &&
                         std::is_same_v<typename TestFixture::BDataType, ck_tile::pk_int4_t>)
            {
                if(K == 2 * TestFixture::K_Tile)
                {
                    // This particular combination of parameters fails.
                    continue;
                }
            }
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s
                constexpr std::string_view suite_name = XSTRINGIFY(TEST_SUITE_NAME);
#undef STRINGIFY
#undef XSTRINGIFY
                if constexpr(suite_name == "TestCkTileGemmPipelineCompTDMWmma")
                {
                    // TestCkTileGemmPipelineCompTDMWmma can handle non-dword-aligned cases
                    this->Run(M, N, K);
                }
                else
                {
#if defined(CK_USE_GFX1250)
                    // On gfx1250 (wave32), float ColA M values satisfying
                    // M % vectorSizeA == 0 are supported and do not throw.
                    // Run those; keep EXPECT_THROW for all other cases.
                    if constexpr(std::is_same_v<typename TestFixture::ADataType, float>)
                    {
                        constexpr int vectorSizeA = 4; // 128-bit load / sizeof(float)
                        if(M % vectorSizeA == 0)
                            this->Run(M, N, K);
                        else
                            EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                    }
                    else
                    {
                        EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                    }
#else
                    EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
#endif
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;

    std::vector<int> Ks;
    for(auto K_count : {2, 3, 4, 10, 11})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }
    constexpr int VecLoadSize = (std::is_same_v<typename TestFixture::ADataType, ck_tile::fp8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::bf8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::int8_t>)
                                    ? 16
                                    : 8;

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ADataType, ck_tile::fp16_t> &&
                         std::is_same_v<typename TestFixture::BDataType, ck_tile::pk_int4_t>)
            {
                if(K == 2 * TestFixture::K_Tile)
                {
                    // This particular combination of parameters fails.
                    continue;
                }
            }
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                if(M % VecLoadSize == 0)
                {
                    this->Run(M, N, K);
                }
                else
                {
#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s
                    constexpr std::string_view suite_name = XSTRINGIFY(TEST_SUITE_NAME);
#undef STRINGIFY
#undef XSTRINGIFY
                    if constexpr(suite_name == "TestCkTileGemmPipelineCompTDMWmma")
                    {
                        // TestCkTileGemmPipelineCompTDMWmma can handle non-dword-aligned cases
                        this->Run(M, N, K);
                    }
                    else
                    {
                        EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                    }
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, PaddK)
{
    std::vector<int> Ms{128};
    constexpr int N = 1024;
    // K=432 is not divisible by 32 (vectorSizeB for pk_int4 on gfx1250 wave32),
    // causing IsSupportedArgument to fail on gfx1250. Use K=416 instead:
    // 416 = 13*32 (divisible by vectorSizeB=32) and 416%64=32 (exercises K-padding).
#if defined(CK_USE_GFX1250)
    constexpr int K_pk_int4 = 416;
#else
    constexpr int K_pk_int4 = 432;
#endif
    constexpr int K = 432;

    if constexpr(TestFixture::PipelineType == GemmPipelineType::CompAsyncEightWaves)
        return;

    for(int M : Ms)
    {
        if constexpr(std::is_same_v<typename TestFixture::BDataType, ck_tile::pk_int4_t>)
        {
#if defined(ARCH_GFX12) || defined(ARCH_GFX11)
            this->Run(M, N, K_pk_int4);
#else
            EXPECT_THROW(this->Run(M, N, K_pk_int4), std::runtime_error);
#endif
        }
        else
        {
            this->Run(M, N, K);
        }
    }
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

#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

    constexpr std::string_view suite_name = XSTRINGIFY(TEST_SUITE_NAME);
    if constexpr(suite_name == "TestCkTileGemmPipelineCompTDMWmma")
    {
        using ADataType = TestFixture::ADataType;
        using BDataType = TestFixture::BDataType;

        constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
        if constexpr((APackedSize == 1) && (BPackedSize == 1))
        {
            this->template Run<PadM, PadN, PadK>(M, N, K);
        }
        else
        {
            GTEST_SKIP()
                << "TestCkTileGemmPipelineCompTDMWmma only supports A and B packed size of 1";
        }
    }
    else
    {
        EXPECT_THROW((this->template Run<PadM, PadN, PadK>(M, N, K)), std::runtime_error);
    }

#undef STRINGIFY
#undef XSTRINGIFY
}
