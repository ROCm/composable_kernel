
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_bwd.hpp"
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = layernorm2d_bwd_args;

template <typename Traits_>
float layernorm2d_bwd_(const S& s, A a)
{
    using DataType = typename Traits_::DataType;

    using PipelineProblem = ck_tile::Layernorm2dBwdGammaBetaPipelineProblem<
        typename LayerNormTypeConfig<DataType>::XDataType,
        typename LayerNormTypeConfig<DataType>::GammaDataType,
        typename LayerNormTypeConfig<DataType>::BetaDataType,
        typename LayerNormTypeConfig<DataType>::ComputeDataType,
        typename LayerNormTypeConfig<DataType>::YDataType,
        typename LayerNormTypeConfig<DataType>::MeanDataType,
        typename LayerNormTypeConfig<DataType>::InvStdDataType,
        typename Traits_::Shape,
        Traits_::kPadN>;

    using Pipeline = ck_tile::Layernorm2dBwdGammaBetaPipeline<PipelineProblem>;

    using Kernel = ck_tile::Layernorm2dBwdGammaBeta<Pipeline>;

    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
