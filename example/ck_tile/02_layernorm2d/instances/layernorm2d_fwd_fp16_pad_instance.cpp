
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"

template <typename Traits_>
float layernorm2d_fwd_(const ck_tile::stream_config& s, layernorm2d_fwd_args a)
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

template <ck_tile::index_t NRepeat,
          ck_tile::index_t NThread,
          ck_tile::index_t VectorAccessSize,
          bool kTwoPass>
using t = layernorm2d_fwd_traits_<ck_tile::fp16_t,
                                  NRepeat,
                                  NThread,
                                  VectorAccessSize,
                                  true,
                                  false,
                                  kTwoPass>;

using S = const ck_tile::stream_config;
using A = layernorm2d_fwd_args;

// Disable all vector 8fp16 read/write instances as it has performance issue regarding compiler
// template float layernorm2d_fwd_<t<1, 16, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<1, 32, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<1, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<2, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<4, 64, 8, false>>(const S&, A);
// template float layernorm2d_fwd_<t<4, 64, 8, true>>(const S&, A);

template float layernorm2d_fwd_<t<1, 32, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<1, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 4, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 4, true>>(const S&, A);

template float layernorm2d_fwd_<t<1, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<2, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<4, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<8, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 64, 2, false>>(const S&, A);
template float layernorm2d_fwd_<t<16, 64, 2, true>>(const S&, A);

template float layernorm2d_fwd_<t<32, 64, 1, false>>(const S&, A);
template float layernorm2d_fwd_<t<32, 64, 1, true>>(const S&, A);
