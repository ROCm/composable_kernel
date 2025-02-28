// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dBwdGammaBetaPipelineDefaultPolicy>
struct Layernorm2dBwdDGammaBetaPipeline
{
    using Problem      = ck_tile::remove_cvref_t<Problem_>;
    using Policy       = ck_tile::remove_cvref_t<Policy_>;
    using ReducePolicy = ck_tile::remove_cvref_t<BlockReduce2dDefaultPolicy>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = Problem::kPadN;

    static constexpr const char* name = []() { return "bwd_gamma_beta"; }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // return ReducePolicy::template GetSmemSize<Problem>();
        using y_block_tile = decltype(make_static_distributed_tensor<ComputeDataType>(Policy::template MakeGammaBetaBlockTileDistribution<Problem>()));
        return ReducePolicy::template GetBlockReduce2dCrossWarpSync<Problem>().template GetSmemSize<y_block_tile>();
        // constexpr index_t smem_size = ReducePolicy::template GetBlockReduce2dCrossWarpSync<Problem>().template GetSmemSize<y_block_tile>();
        // static_assert(smem_size==0, "smem_size not correct");
        // return smem_size;
    }

    template <typename XWindow,
              typename YWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename DGammaWindow,
              typename DBetaWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const YWindow& dy_window_,
                                   const MeanWindow& mean_window_,
                                   const InvStdWindow& inv_std_window_,
                                   DGammaWindow& dgamma_window_,
                                   DBetaWindow& dbeta_window_,
                                   ck_tile::index_t column_size,
                                   void* smem) const
    {

        auto dgamma_beta_dist = Policy::template MakeGammaBetaBlockTileDistribution<Problem>();
        auto mean_dist        = Policy::template MakeMeanBlockTileDistribution<Problem>();
        auto x_dist           = Policy::template MakeXBlockTileDistribution<Problem>();

        // const auto x_window       = make_tile_window(x_window_, x_dist);
        // const auto dy_window      = make_tile_window(dy_window_, x_dist);
        // const auto mean_window    = make_tile_window(mean_window_, mean_dist);
        // const auto inv_std_window = make_tile_window(inv_std_window_, mean_dist);
        auto x_window       = make_tile_window(x_window_, x_dist);
        auto dy_window      = make_tile_window(dy_window_, x_dist);
        auto mean_window    = make_tile_window(mean_window_, mean_dist);
        auto inv_std_window = make_tile_window(inv_std_window_, mean_dist);

        auto dgamma_window = make_tile_window(dgamma_window_, dgamma_beta_dist);
        auto dbeta_window  = make_tile_window(dbeta_window_, dgamma_beta_dist);

        // const auto x_tile         = load_tile(x_window);
        // const auto dy_tile        = load_tile(dy_window);
        // const auto mean_tile      = load_tile(mean_window);
        // const auto inv_std_tile   = load_tile(inv_std_window);

        auto dgamma_tile = make_static_distributed_tensor<GammaDataType>(dgamma_beta_dist);
        auto dbeta_tile  = make_static_distributed_tensor<BetaDataType>(dgamma_beta_dist);
        auto dgamma      = cast_tile<ComputeDataType>(dgamma_tile);
        auto dbeta       = cast_tile<ComputeDataType>(dbeta_tile);

        static constexpr index_t Block_M = Problem::BlockShape::Block_M;
        index_t num_m_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(column_size, Block_M));

        for(int iM = __builtin_amdgcn_readfirstlane(0); iM < num_m_tile_iteration; ++iM)
        {
            const auto x_tile         = load_tile(x_window);
            const auto dy_tile        = load_tile(dy_window);
            const auto mean_tile      = load_tile(mean_window);
            const auto inv_std_tile   = load_tile(inv_std_window);

            move_tile_window(x_window, {Block_M, 0});
            move_tile_window(dy_window, {Block_M, 0});
            move_tile_window(mean_window, {Block_M});
            move_tile_window(inv_std_window, {Block_M});

            sweep_tile(x_tile, [&](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);
                const auto x         = type_convert<ComputeDataType>(x_tile[idx]);
                const auto dy        = type_convert<ComputeDataType>(dy_tile[idx]);
                const auto mean      = type_convert<ComputeDataType>(mean_tile[i_idx]);
                const auto inv_std   = type_convert<ComputeDataType>(inv_std_tile[i_idx]);
                dbeta(j_idx) += dy;
                dgamma(j_idx) += dy * (x - mean) * inv_std;
                printf("dy: threadidx=%d, blockidx=%d, x_tile=%f\n",threadIdx.x, blockIdx.x, dy);
            });
        }

        auto block_reduce2d_sync = ReducePolicy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync = ReducePolicy::template GetBlockReduce2dCrossWarpSync<Problem>();
        block_reduce2d_sync(dbeta, ck_tile::ReduceOp::Add{});
        block_reduce2d_sync(dgamma, ck_tile::ReduceOp::Add{});
        // sweep_tile(dbeta, [&](auto idx) {
        //     printf("dbeta pre: warpid=%d, threadidx=%d, blockidx=%d, dbeta=%f\n", get_warp_id(), threadIdx.x, blockIdx.x,
        //     dbeta[idx]);
        // });
        block_reduce2d_cross_warp_sync(dbeta, smem, ck_tile::ReduceOp::Add{});
        block_reduce2d_cross_warp_sync(dgamma, smem, ck_tile::ReduceOp::Add{});

        // sweep_tile(dbeta, [&](auto idx) {
        //     printf("dbeta post: warpid=%d, threadidx=%d, blockidx=%d, dbeta=%f\n", get_warp_id(), threadIdx.x, blockIdx.x,
        //     dbeta[idx]);
        // });

        store_tile(dbeta_window, cast_tile<BetaDataType>(dbeta));
        store_tile(dgamma_window, cast_tile<GammaDataType>(dgamma));
    }
};
} // namespace ck_tile
