// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "layernorm2d_fwd.hpp"

template <typename InOutDataType,
          ck_tile::index_t NRepeat,
          ck_tile::index_t NThread,
          ck_tile::index_t VectorAccessSize,
          bool kPadN,
          bool kTwoPass>
struct layernorm_dispatch
{
    static constexpr ck_tile::index_t MRepeat = 1;
    static_assert(NThread <= 64, "We only support intra-wave reduction");
    static constexpr ck_tile::index_t WaveNum = NThread / 16;
    // clang-format off
    using thread_tile = ck_tile::sequence<MRepeat, NRepeat, VectorAccessSize>;
    using warp_tile   = ck_tile::sequence<MRepeat*64/NThread, NRepeat * NThread*VectorAccessSize>;
    using block_tile  = ck_tile::sequence<MRepeat*WaveNum*64/NThread, NRepeat * NThread*VectorAccessSize>;
    // clang-format on

    using Shape = ck_tile::TileLayernorm2dShape<thread_tile, warp_tile, block_tile>;

    using PipelineProblem = ck_tile::BlockLayernorm2dFwdProblem<
        typename LayerNormTypeConfig<InOutDataType>::XDataType,
        typename LayerNormTypeConfig<InOutDataType>::GammaDataType,
        typename LayerNormTypeConfig<InOutDataType>::BetaDataType,
        typename LayerNormTypeConfig<InOutDataType>::ComputeDataType,
        typename LayerNormTypeConfig<InOutDataType>::YDataType,
        typename LayerNormTypeConfig<InOutDataType>::MeanDataType,
        typename LayerNormTypeConfig<InOutDataType>::InvStdDataType,
        Shape,
        kPadN,
        kTwoPass>;

    using Kernel = ck_tile::Layernorm2dFwd<PipelineProblem>;

    static float Run(const layernorm2d_fwd_args& param, ck_tile::stream_config stream)
    {
        using k_ = Kernel;

        const dim3 grids                       = k_::GridSize(param.M);
        constexpr dim3 blocks                  = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        return ck_tile::launch_kernel(stream,
                                      ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{},
                                                                                  grids,
                                                                                  blocks,
                                                                                  0,
                                                                                  param.p_x,
                                                                                  param.p_gamma,
                                                                                  param.p_beta,
                                                                                  param.p_y,
                                                                                  param.epsilon,
                                                                                  param.M,
                                                                                  param.N));
    };
};

template <typename InOutDataType,
          ck_tile::index_t NRepeat,
          ck_tile::index_t NThread,
          ck_tile::index_t VectorAccessSize,
          bool kPadN,
          bool kTwoPass = false>
float run_layernorm(const layernorm2d_fwd_args& param, ck_tile::stream_config stream)
{
    return layernorm_dispatch<InOutDataType, NRepeat, NThread, VectorAccessSize, kPadN, kTwoPass>::
        Run(param, stream);
};
