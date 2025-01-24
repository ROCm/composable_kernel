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
struct Layernorm2dBwdGammaBetaPipelineTwoPass
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
        return ReducePolicy::template GetSmemSize<Problem>();
        //GetBlockReduce2dCrossWarpSync<Problem>().template GetSmemSize<y_block_tile>();
    }
    template <typename XWindow,
              typename GammaWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename DGammaWindow,
              typename DBetaWindow,
              typename DXWindow,

              // tmp
              typename DSWindow,
              typename DBWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XWindow& dy_window_,
                                   const GammaWindow& gamma_window_,
                                   const MeanWindow& mean_window_,
                                   const InvStdWindow& inv_std_window_,
                                   DGammaWindow& dgamma_window_,
                                   DBetaWindow& dbeta_window_,
                                   DXWindow& dx_window_,

                                   // tmp
                                   DSWindow& ds_window_,
                                   DBWindow& db_window_,

                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        
        auto gamma_beta_dist  = Policy::template MakeGammaBetaBlockTileDistribution<Problem>();
        auto dgamma_beta_dist = Policy::template MakeDGammaBetaBlockTileDistribution<Problem>();
        auto mean_dist        = Policy::template MakeMeanBlockTileDistribution<Problem>();
        auto x_dist           = Policy::template MakeXBlockTileDistribution<Problem>();

        auto x_window       = make_tile_window(x_window_, x_dist);
        auto dy_window      = make_tile_window(dy_window_, x_dist);
        auto gamma_window   = make_tile_window(gamma_window_, gamma_beta_dist); // TO CHECK
        auto mean_window    = make_tile_window(mean_window_, mean_dist);
        auto inv_std_window = make_tile_window(inv_std_window_, mean_dist);

        auto dgamma_window = make_tile_window(dgamma_window_, dgamma_beta_dist);
        auto dbeta_window  = make_tile_window(dbeta_window_, dgamma_beta_dist);
        auto dx_window     = make_tile_window(dx_window_, x_dist);

        const auto mean_tile      = load_tile(mean_window);
        const auto inv_std_tile   = load_tile(inv_std_window);

        // tmp
        (void)ds_window_;
        (void)db_window_;
        //auto ds_window = make_tile_window(ds_window_, mean_dist);
        //auto db_window = make_tile_window(db_window_, mean_dist);
        auto ds_tile   = make_static_distributed_tensor<ComputeDataType>(mean_dist);
        auto db_tile   = make_static_distributed_tensor<ComputeDataType>(mean_dist);
        clear_tile(ds_tile);
        clear_tile(db_tile);

        // auto dgamma_tile = make_static_distributed_tensor<GammaDataType>(dgamma_beta_dist);
        // auto dbeta_tile  = make_static_distributed_tensor<BetaDataType>(dgamma_beta_dist);
        auto dx_tile     = make_static_distributed_tensor<XDataType>(x_dist);
        // auto dgamma      = cast_tile<ComputeDataType>(dgamma_tile);
        // auto dbeta       = cast_tile<ComputeDataType>(dbeta_tile);
        auto dx          = cast_tile<ComputeDataType>(dx_tile);

        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_tile         = load_tile(x_window);
            const auto dy_tile        = load_tile(dy_window);
            const auto gamma_tile     = load_tile(gamma_window);

            move_tile_window(x_window, {0, Block_N});
            move_tile_window(dy_window, {0, Block_N});
            move_tile_window(gamma_window, {Block_N});

            sweep_tile(x_tile, [&](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);
                const auto x         = type_convert<ComputeDataType>(x_tile[idx]);
                const auto dy        = type_convert<ComputeDataType>(dy_tile[idx]);
                const auto gamma     = type_convert<ComputeDataType>(gamma_tile[j_idx]);
                ds_tile(i_idx) += dy * gamma * x;
                db_tile(i_idx) += dy * gamma;
                // dx(idx) = dy * gamma;
                // ds_tile(i_idx) += dx[idx] * x;
                // db_tile(i_idx) += dx[idx];
            // printf("db_tile pre: threadidx=%d, blockidx=%d, db_tile=%f\n",threadIdx.x, blockIdx.x, db_tile[i_idx]);
            // printf("dy_tile: threadidx=%d, blockidx=%d, dy_tile=%f\n",threadIdx.x, blockIdx.x, dy);
            // printf("x: threadidx=%d, blockidx=%d, x_tile=%f\n",threadIdx.x, blockIdx.x, x);
            // printf("gamma: threadidx=%d, blockidx=%d, gamma_tile=%f\n",threadIdx.x, blockIdx.x, gamma);
            });
        }

        auto block_reduce2d_sync = ReducePolicy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync = ReducePolicy::template GetBlockReduce2dCrossWarpSync<Problem>();
        block_reduce2d_sync(ds_tile, ck_tile::ReduceOp::Add{});
        block_reduce2d_sync(db_tile, ck_tile::ReduceOp::Add{});
        // block_reduce2d_cross_warp_sync(ds_tile, smem, ck_tile::ReduceOp::Add{});
        // block_reduce2d_cross_warp_sync(db_tile, smem, ck_tile::ReduceOp::Add{});

        // sweep_tile(x_tile, [&](auto idx) {
        //     constexpr auto i_idx = make_tuple(idx[number<0>{}]);
        //     printf("db_tile post: threadidx=%d, blockidx=%d, db_tile=%f\n",threadIdx.x, blockIdx.x,
        //     db_tile[i_idx]);
        // });

        // store_tile(ds_window, ds_tile);
        // store_tile(db_window, db_tile);

        ck_tile::index_t stride_to_right_most_window = row_size % Block_N == 0 ? row_size - Block_N : row_size - row_size % Block_N;
        move_tile_window(x_window, {0, -Block_N});
        move_tile_window(dy_window, {0, -Block_N});
        move_tile_window(gamma_window, {-Block_N});
        move_tile_window(dx_window, {0, stride_to_right_most_window});
        // move_tile_window(dbeta_window, {0, stride_to_right_most_window});
        // move_tile_window(dgamma_window, {0, stride_to_right_most_window});

        using XDistributedTensor = decltype(load_tile(x_window));
        constexpr auto spans     = XDistributedTensor::get_distributed_spans();

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_tile         = load_tile(x_window);
            const auto dy_tile        = load_tile(dy_window);
            const auto gamma_tile     = load_tile(gamma_window);
            sweep_tile_span(spans[number<0>{}], [&](auto i_idx) {
                constexpr auto idx0 = make_tuple(i_idx);
                const auto mean     = type_convert<ComputeDataType>(mean_tile[idx0]);
                const auto inv_std  = type_convert<ComputeDataType>(inv_std_tile[idx0]);
                auto b = (db_tile[idx0] * mean - ds_tile[idx0]) * inv_std * inv_std * inv_std / row_size;
                auto c = -b * mean - db_tile[idx0] * inv_std / row_size;

                sweep_tile_span(spans[number<1>{}], [&](auto j_idx) {
                constexpr auto idx1   = make_tuple(j_idx);
                    constexpr auto idx    = make_tuple(i_idx, j_idx);
                    //constexpr auto gb_idx = make_tuple(number<0>{}, j_idx);
                    const auto x          = type_convert<ComputeDataType>(x_tile[idx]);
                    const auto dy         = type_convert<ComputeDataType>(dy_tile[idx]);
                    const auto gamma      = type_convert<ComputeDataType>(gamma_tile[idx1]);
                    // dbeta(gb_idx) += dy;
                    // dgamma(gb_idx) += dy * (x - mean) * inv_std;
                    dx(idx) = dy * gamma * inv_std + b * x + c;
                    //printf("dx: threadidx=%d, blockidx=%d, dx_tile=%f\n",threadIdx.x, blockIdx.x, dx(idx));
                });
            });
            // store_tile(dbeta_window, cast_tile<BetaDataType>(dbeta));
            // store_tile(dgamma_window, cast_tile<GammaDataType>(dgamma));
            store_tile(dx_window, cast_tile<XDataType>(dx));

            move_tile_window(x_window, {0, -Block_N});
            move_tile_window(dy_window, {0, -Block_N});
            move_tile_window(gamma_window, {-Block_N});
            move_tile_window(dx_window, {0, -Block_N});
            // move_tile_window(dbeta_window, {0, -Block_N});
            // move_tile_window(dgamma_window, {0, -Block_N});
        }
    }
};
} // namespace ck_tile
