
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"

#pragma once

using S = ck_tile::stream_config;
using A = layernorm2d_fwd_args;

template <typename Traits_>
float layernorm2d_fwd_(const S& s, A a)
{
    using DataType = typename Traits_::DataType;

    using PipelineProblem =
        ck_tile::BlockLayernorm2dFwdProblem<typename LayerNormTypeConfig<DataType>::XDataType,
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

    using Kernel = ck_tile::Layernorm2dFwd<PipelineProblem>;

    const dim3 grids                       = Kernel::GridSize(a.M);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    return ck_tile::launch_kernel(s,
                                  ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{},
                                                                              grids,
                                                                              blocks,
                                                                              0,
                                                                              a.p_x,
                                                                              a.p_gamma,
                                                                              a.p_beta,
                                                                              a.p_y,
                                                                              a.p_mean,
                                                                              a.p_invStd,
                                                                              a.epsilon,
                                                                              a.M,
                                                                              a.N));
}
