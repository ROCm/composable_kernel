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
                if constexpr(TestFixture::IsTdm)
                {
                    // TDM pipelines can handle non-dword-aligned cases
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

// K smaller than one K_Tile: the main loop runs a single, partially populated
// iteration. Nothing else reaches this shape -- SmallM/MidLargeM start at
// 2*K_Tile, SingleTile is exactly K_Tile, and NotSupportedArgument's K=513 is a
// tail on top of eight full tiles. TDM pipelines clip on real descriptor
// extents, so they must produce correct results here rather than reading past
// the end of A/B.
TYPED_TEST(TEST_SUITE_NAME, PartialKTile)
{
    if constexpr(TestFixture::IsTdm && TestFixture::check_data_type())
    {
        using ADataType = typename TestFixture::ADataType;
        using BDataType = typename TestFixture::BDataType;
        if constexpr(ck_tile::numeric_traits<ADataType>::PackedSize != 1 ||
                     ck_tile::numeric_traits<BDataType>::PackedSize != 1)
        {
            GTEST_SKIP() << "TDM pipelines only support A and B packed size of 1";
        }
        else
        {
            constexpr int N = 1024;
            // Half a tile, a ragged extent one short of a full tile, and the
            // smallest non-trivial contraction.
            for(int K : {TestFixture::K_Tile / 2, TestFixture::K_Tile - 1, 1})
            {
                if(K <= 0)
                    continue;
                this->Run(TestFixture::M_Tile, N, K);
            }
        }
    }
    else
    {
        GTEST_SKIP() << "K < K_Tile is a TDM-only contract (other pipelines pad K).";
    }
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
                    if constexpr(TestFixture::IsTdm)
                    {
                        // TDM pipelines can handle non-dword-aligned cases
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

// Regression guard for the >2GB single-dimension byte extent rejection in
// UniversalGemmKernel::IsSupportedArgument.
//
// The guard used to sit *below* the `skipCheckValidLaunchParams` short-circuit, so every
// pipeline that sets that trait (all three TDM pipelines) accepted operands whose byte
// extent overflows the 32-bit buffer offset arithmetic and faulted at run time. Nothing
// covered that, because the trigger cannot be allocated: a rejecting shape needs a >2GB
// tensor. This test therefore validates the *arguments* only, never launching, so it
// costs no device memory and runs everywhere.
//
// The first assertion is the control: it proves a shape just under the limit is accepted,
// so a `false` in the second assertion can only have come from the size guard and not
// from an unrelated early rejection (e.g. the gfx1250 named-barrier check).
TYPED_TEST(TEST_SUITE_NAME, LargeTensorArgumentRejected)
{
    // Same compile-time gate Run() applies: type combinations with no WMMA/MFMA traits
    // do not instantiate, and instantiating one here would break the whole TU rather
    // than fail a single case.
    if constexpr(!TestFixture::check_data_type())
    {
        GTEST_SKIP() << "Unsupported data type combination for gemm pipeline test.";
    }
    else
    {
        using ADataType = typename TestFixture::ADataType;
        using BDataType = typename TestFixture::BDataType;
        using CDataType = typename TestFixture::CDataType;

        // The guard compares n * stride * sizeof(T) / PackedSize against 2^31. Keep
        // sizeof and PackedSize separate: for a packed type (pk_int4_t, pk_fp4_t)
        // sizeof(T) / PackedSize integer-divides to zero.
        constexpr size_t kSizeLimit = size_t{1} << 31;

        constexpr size_t a_bytes = sizeof(ADataType);
        constexpr size_t b_bytes = sizeof(BDataType);
        constexpr size_t c_bytes = sizeof(CDataType);
        constexpr size_t a_pack  = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr size_t b_pack  = ck_tile::numeric_traits<BDataType>::PackedSize;
        constexpr size_t c_pack  = ck_tile::numeric_traits<CDataType>::PackedSize;

        // A modest, tile-aligned shape. Only the strides are grown, so the guard is the only
        // thing that changes between the accepted and the rejected case.
        const int M = TestFixture::M_Tile;
        const int N = TestFixture::N_Tile;
        const int K = TestFixture::K_Tile;

        const auto row_major = [](auto layout) {
            return std::is_same_v<ck_tile::remove_cvref_t<decltype(layout)>,
                                  ck_tile::tensor_layout::gemm::RowMajor>;
        };

        // Leading dimension of each operand as the guard sees it: rows for RowMajor, cols
        // for ColumnMajor.
        const size_t a_n = row_major(typename TestFixture::ALayout{}) ? size_t(M) : size_t(K);
        const size_t b_n = row_major(typename TestFixture::BLayout{}) ? size_t(K) : size_t(N);
        const size_t c_n = row_major(typename TestFixture::CLayout{}) ? size_t(M) : size_t(N);

        // Stride that makes the guard's byte extent land on `bytes`:
        //   bytes = n * stride * elem_bytes / packed  =>  stride = bytes * packed / (n *
        //   elem_bytes)
        const auto stride_at = [](size_t bytes, size_t n, size_t elem_bytes, size_t packed) {
            return static_cast<ck_tile::index_t>(bytes * packed / (n * elem_bytes));
        };

        constexpr size_t kUnderLimit = kSizeLimit - kSizeLimit / 4;

        // Just under 2GB: must be accepted.
        const ck_tile::index_t sa_ok = stride_at(kUnderLimit, a_n, a_bytes, a_pack);
        const ck_tile::index_t sb_ok = stride_at(kUnderLimit, b_n, b_bytes, b_pack);
        const ck_tile::index_t sc_ok = stride_at(kUnderLimit, c_n, c_bytes, c_pack);

        EXPECT_TRUE(TestFixture::IsArgSupported(M, N, K, sa_ok, sb_ok, sc_ok))
            << "sub-2GB extents must be accepted; a later assertion is meaningless otherwise";

        // At or above 2GB in exactly one operand at a time: each must be rejected on its own.
        EXPECT_FALSE(TestFixture::IsArgSupported(
            M, N, K, stride_at(kSizeLimit, a_n, a_bytes, a_pack) + 1, sb_ok, sc_ok))
            << "A with a >=2GB byte extent must be rejected without the LargeTensors trait";
        EXPECT_FALSE(TestFixture::IsArgSupported(
            M, N, K, sa_ok, stride_at(kSizeLimit, b_n, b_bytes, b_pack) + 1, sc_ok))
            << "B with a >=2GB byte extent must be rejected without the LargeTensors trait";
        EXPECT_FALSE(TestFixture::IsArgSupported(
            M, N, K, sa_ok, sb_ok, stride_at(kSizeLimit, c_n, c_bytes, c_pack) + 1))
            << "C with a >=2GB byte extent must be rejected without the LargeTensors trait";
    }
}

TYPED_TEST(TEST_SUITE_NAME, NotSupportedArgument)
{
    constexpr int M = 512;
    constexpr int N = 1025;
    constexpr int K = 513;

    constexpr bool PadM = false;
    constexpr bool PadN = false;
    constexpr bool PadK = false;

    if constexpr(TestFixture::IsTdm)
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
            GTEST_SKIP() << "TDM pipelines only support A and B packed size of 1";
        }
    }
    else
    {
        EXPECT_THROW((this->template Run<PadM, PadN, PadK>(M, N, K)), std::runtime_error);
    }
}
