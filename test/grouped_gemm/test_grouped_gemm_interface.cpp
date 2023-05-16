// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "test_grouped_gemm_util.hpp"

class TestGGemmSplitKInterface_MKNK : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using ALayout = Row;
    using BLayout = Col;
    using ELayout = Row;

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

    template <ck::index_t KPerBlock,
              ck::index_t K1,
              ck::index_t ABlockTransferSrcScalarPerVector,
              ck::index_t BBlockTransferSrcScalarPerVector,
              ck::index_t CDEBlockTransferScalarPerVector_NPerBlock>
    using GGemmInstance =
        ck::test::DeviceGroupedGemmSplitkInstanceWrapper<ALayout,
                                                         BLayout,
                                                         ELayout,
                                                         GemmDefault,
                                                         KPerBlock,
                                                         K1,
                                                         ABlockTransferSrcScalarPerVector,
                                                         BBlockTransferSrcScalarPerVector,
                                                         CDEBlockTransferScalarPerVector_NPerBlock>;

    using DefaultGGemmInstance = GGemmInstance<32, 8, 4, 8, 8>;
};

TEST_F(TestGGemmSplitKInterface_MKNK, TileSize)
{
    std::vector<int> Ms{128, 256, 188, 512};
    int N = 256;
    int K = 128;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // M % MPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ns = std::vector<int>{256, 177, 128, 512};
    // N % NPerBlock
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}

TEST_F(TestGGemmSplitKInterface_MKNK, VectorLoadWidth)
{
    std::vector<int> Ms{128, 256, 256, 512};
    int N = 256;
    int K = 512;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // K % ABlockTransferSrcScalarPerVector
    Ks = std::vector<int>{256, 177, 128, 512};
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ks = std::vector<int>{256, 164, 128, 512};
    // K % BBlockTransferSrcScalarPerVector
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));

    Ks = std::vector<int>(4, 128);
    Ns = std::vector<int>{256, 153, 128, 512};
    // N % CBlockTransferScalarPerVector_NWaveNPerXDL
    EXPECT_FALSE(DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs));
}

TEST_F(TestGGemmSplitKInterface_MKNK, KLoops)
{
    std::vector<int> Ms{128, 256, 256, 512};
    int N      = 256;
    int K      = 128;
    int kbatch = 4;

    std::vector<int> Ns(Ms.size(), N);
    std::vector<int> Ks(Ms.size(), K);
    std::vector<int> StrideAs(Ms.size(), K);
    std::vector<int> StrideBs(Ms.size(), K);
    std::vector<int> StrideCs(Ms.size(), N);

    // kloops % 2
    Ks = std::vector<int>{256, 512, 320, 768};
    EXPECT_FALSE(
        DefaultGGemmInstance{}.IsSupported(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch));

    // Not all gemms have same value for main_k0_block_loop!
    Ks = std::vector<int>{256, 512, 512, 512};
    EXPECT_THROW(DefaultGGemmInstance{}.Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch),
                 std::runtime_error);
}
