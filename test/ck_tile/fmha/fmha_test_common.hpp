// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

namespace ck_tile::test {

template <int HeadDim,
          bool Group    = false,
          bool Mask     = false,
          bool Lse      = false,
          typename Data = ck_tile::bf16_t,
          int N         = 128,
          int ValueDim  = 128>
struct Model
{
    using Shape  = ck_tile::TileFmhaShape<ck_tile::sequence<128, N, 32, ValueDim, 32, HeadDim>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 32>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 32>,
                                          true>;
    using Traits = ck_tile::TileFmhaTraits<true,
                                           true,
                                           true,
                                           true,
                                           false,
                                           ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                           false,
                                           Lse,
                                           false,
                                           ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                           1,
                                           false,
                                           false>;
    using Problem =
        ck_tile::BlockFmhaPipelineProblem<Data,
                                          Data,
                                          Data,
                                          float,
                                          float,
                                          Data,
                                          std::uint8_t,
                                          float,
                                          Data,
                                          float,
                                          Data,
                                          Shape,
                                          Group,
                                          ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                          ck_tile::SimplifiedGenericAttentionMask<Mask>,
                                          false,
                                          Traits>;
    static_assert(Traits::kStoreLSE == Lse && !Traits::kHasBiasGrad);
    static_assert(Problem::kStoreLSE == Lse);
    using Geometry = ck_tile::FmhaTdmSchedGeometryFor<Problem>;
};

// Split-kv counterpart of Model: same shape and policy as the prefill qr_tdm,
// with the split-kv problem type. The flags pin one point of the trait space;
// BlockM selects the tile (split-kv qr_tdm runs the 64-wide one).
template <int HeadDim,
          bool Mask         = false,
          typename Data     = ck_tile::bf16_t,
          int N             = 128,
          int ValueDim      = 128,
          bool UnevenSplits = true,
          int BlockM        = 128>
struct SplitKVModel
{
    using Shape   = ck_tile::TileFmhaShape<ck_tile::sequence<BlockM, N, 32, ValueDim, 32, HeadDim>,
                                           ck_tile::sequence<4, 1, 1>,
                                           ck_tile::sequence<16, 16, 32>,
                                           ck_tile::sequence<4, 1, 1>,
                                           ck_tile::sequence<16, 16, 32>,
                                           true>;
    using Traits  = ck_tile::TileFmhaFwdSplitKVTraits<true,  // kPadSeqLenQ
                                                      true,  // kPadSeqLenK
                                                      false, // kPadHeadDimQ
                                                      false, // kPadHeadDimV
                                                      false, // kHasLogitsSoftCap
                                                      ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                      false, // kHasBiasGrad
                                                      true,  // kStoreLSE
                                                      false, // kDoFp8StaticQuant
                                                      false, // kIsPagedKV
                                                      UnevenSplits,
                                                      false,  // kMergeNumHeadGroupsSeqLenQ
                                                      1,      // kBlockPerCu
                                                      false>; // kHasSink
    using Problem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
        Data,
        Data,
        Data,
        float,
        float,
        Data,
        float,
        Data,
        float,
        float,
        Shape,
        false,
        ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
        ck_tile::SimplifiedGenericAttentionMask<Mask>,
        Traits>;
    using TdmPipeline = ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSTdm<Problem>;
};

class Gfx125FmhaTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        int count         = 0;
        const auto status = hipGetDeviceCount(&count);
        if(status == hipErrorNoDevice || (status == hipSuccess && count == 0))
            GTEST_SKIP() << "No HIP device available";
        ASSERT_EQ(status, hipSuccess);
        int device = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
        if(std::string_view{properties.gcnArchName}.find("gfx125") != 0)
            GTEST_SKIP() << "qr_tdm_sched requires the gfx125 family";
    }
};

template <typename Problem>
using Policy = ck_tile::FmhaTdmSchedPolicyFor<Problem>;

} // namespace ck_tile::test
