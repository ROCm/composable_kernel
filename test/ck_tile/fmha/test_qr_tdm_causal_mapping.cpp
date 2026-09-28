// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "ck_tile/host/device_memory.hpp"

#include "gtest/gtest.h"

#include <vector>

namespace {

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false>
using MappingProblem = ck_tile::BlockFmhaPipelineProblem<
    DataType,
    DataType,
    DataType,
    float,
    float,
    DataType,
    uint8_t,
    float,
    DataType,
    float,
    DataType,
    ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           true>,
    false,
    ck_tile::ComposedAttention<0>,
    ck_tile::SimplifiedGenericAttentionMask<Mask>,
    false,
    ck_tile::TileFmhaTraits<false,
                            false,
                            false,
                            false,
                            false,
                            BiasEnum,
                            false,
                            StoreLSE,
                            false,
                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                            -1,
                            false,
                            HasSink>,
    true,
    true>;

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false>
using MappingKernel = ck_tile::FmhaFwdKernel<
    ck_tile::BlockFmhaPipelineQRKSVSTdm<
        MappingProblem<M, Mask, StoreLSE, DataType, BiasEnum, HasSink>>,
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, DataType, false, false>>>;

template <typename Kernel>
__global__ void capture_mapping(typename Kernel::Kargs args, int* output)
{
    if(threadIdx.x == 0)
    {
        const auto [q, v, head, batch] = Kernel::GetTileIndex(args);
        const auto linear      = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        output[linear * 4 + 0] = q;
        output[linear * 4 + 1] = v;
        output[linear * 4 + 2] = head;
        output[linear * 4 + 3] = batch;
    }
}

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false>
std::vector<int> capture(int heads,
                         int batches,
                         int tiles,
                         int left    = -1,
                         int extra_k = 0,
                         int hdim_q  = 128,
                         int hdim_v  = 128)
{
    using Kernel = MappingKernel<M, Mask, StoreLSE, DataType, BiasEnum, HasSink>;
    typename Kernel::Kargs args{};
    args.hdim_q         = hdim_q;
    args.hdim_v         = hdim_v;
    args.seqlen_q       = tiles * M;
    args.seqlen_k       = tiles * M + extra_k;
    args.stride_q       = hdim_q;
    args.nhead_stride_q = args.seqlen_q * hdim_q;
    if constexpr(Mask)
    {
        args.window_size_left  = left;
        args.window_size_right = 0;
        args.mask_type         = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
    }
    const auto grid = Kernel::GridSize(batches, heads, args.seqlen_q, args.hdim_v);
    std::vector<int> result(grid.x * grid.y * grid.z * 4);
    ck_tile::DeviceMem device(result.size() * sizeof(int));
    hipLaunchKernelGGL(capture_mapping<Kernel>,
                       grid,
                       dim3(32),
                       0,
                       0,
                       args,
                       static_cast<int*>(device.GetDeviceBuffer()));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    device.FromDevice(result.data());
    return result;
}

TEST(QrTdmCausalMapping, CausalLongTilesComeFirstAcrossHeads)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    // A head-major implementation restarts at the longest tile for each head;
    // this observable schedule must instead keep long causal work together.
    const auto actual = capture<128, true>(2, 1, 4);
    const std::vector<int> expected{3, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1, 0,
                                    1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ(actual, expected);
}

TEST(QrTdmCausalMapping, DenseRetainsHeadMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const auto actual = capture<128, false>(2, 1, 3);
    const std::vector<int> expected{0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0,
                                    0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0};
    EXPECT_EQ(actual, expected);
}

TEST(QrTdmCausalMapping, OtherMasksAndRectangularInputsRetainTheirSchedule)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                    2, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<128, true>(2, 1, 3, 127)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 128)), expected);
}

TEST(QrTdmCausalMapping, M64CoversEachBatchHeadAndTileExactlyOnce)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const auto actual = capture<64, true>(8, 2, 6);
    std::vector<int> visits(2 * 8 * 6);
    for(std::size_t i = 0; i < actual.size(); i += 4)
    {
        const int q = actual[i], v = actual[i + 1], h = actual[i + 2], b = actual[i + 3];
        ASSERT_GE(q, 0);
        ASSERT_LT(q, 6);
        ASSERT_EQ(v, 0);
        ASSERT_GE(h, 0);
        ASSERT_LT(h, 8);
        ASSERT_GE(b, 0);
        ASSERT_LT(b, 2);
        ++visits[(b * 8 + h) * 6 + q];
    }
    for(int count : visits)
        EXPECT_EQ(count, 1);
}

TEST(QrTdmCausalMapping, SupportedVariantsUseQMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0,
                                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<128, true, true>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128, true, false, ck_tile::half_t>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128,
                       true,
                       false,
                       ck_tile::bf16_t,
                       ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS>(2, 1, 3)),
              expected);
    EXPECT_EQ(
        (capture<128, true, false, ck_tile::bf16_t, ck_tile::BlockAttentionBiasEnum::NO_BIAS, true>(
            2, 1, 3)),
        expected);
}

TEST(QrTdmCausalMapping, SingleOutputTileVariantsUseQMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0,
                                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<64, true>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 64, 128)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 64)), expected);
}

TEST(QrTdmCausalMapping, MultipleOutputTilesRetainHeadMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                                    0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 2, 1, 1, 0,
                                    1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0};
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 256)), expected);
}

} // namespace
