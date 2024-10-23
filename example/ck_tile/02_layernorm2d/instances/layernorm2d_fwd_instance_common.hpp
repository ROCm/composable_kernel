
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = layernorm2d_fwd_args;

template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kTwoPass_>
using trait_ = layernorm2d_fwd_traits_<DataType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_N_,
                                       kPadN_,
                                       kSaveMeanInvStd_,
                                       kTwoPass_>;

template <typename Traits_>
float layernorm2d_fwd_(const S& s, A a)
{
    using DataType = typename Traits_::DataType;

    using PipelineProblem = ck_tile::Layernorm2dFwdPipelineProblem<
        typename LayerNormTypeConfig<DataType>::XDataType,
        typename LayerNormTypeConfig<DataType>::GammaDataType,
        typename LayerNormTypeConfig<DataType>::BetaDataType,
        typename LayerNormTypeConfig<DataType>::ComputeDataType,
        typename LayerNormTypeConfig<DataType>::YDataType,
        typename LayerNormTypeConfig<DataType>::MeanDataType,
        typename LayerNormTypeConfig<DataType>::InvStdDataType,
        typename Traits_::Shape,
        Traits_::kPadN,
        Traits_::kSaveMeanInvStd,
        Traits_::kTwoPass>;

    using OnePassPipeline = ck_tile::Layernorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using Kernel = ck_tile::Layernorm2dFwd<Pipeline>;

    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
