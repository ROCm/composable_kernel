
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"

#pragma once

#ifndef _MAX2
#define _MAX2(a, b) ((a) > (b) ? (a) : (b))
#endif

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kTwoPass_>
struct layernorm2d_fwd_traits_
{
    using DataType = ck_tile::remove_cvref_t<DataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        }
        else
        {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;

    using Shape = ck_tile::Layernorm2dShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kTwoPass        = kTwoPass_;
};

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
#include <iostream>
template <typename Traits_>
float layernorm2d_fwd_(const S& s, A a)
{
    using DataType = typename Traits_::DataType;

    using PipelineProblem = ck_tile::Layernorm2dFwdRowwiseProblem<
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

    using OnePassPipeline = ck_tile::Layernorm2dFwdOnePassPipeline<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdTwoPassPipeline<PipelineProblem>;
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

#undef _MAX2
