// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dFwdPipelineDefaultPolicy>
struct Layernorm2dFwdPipelineTwoPass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kHasBeta    = !std::is_same_v<BetaDataType, ck_tile::null_type>;
    static constexpr bool kSaveMean   = Problem::kSaveMeanInvStd;
    static constexpr bool kSaveInvStd = Problem::kSaveMeanInvStd;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockLayernorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::kPadN;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_tp"; // block per row
        else
            return "wpr_tp"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename GammaWindow,
              typename BetaWindow,
              typename YWindow,
              typename MeanWindow,
              typename InvStdWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const GammaWindow& gamma_window_,
                                   const BetaWindow& beta_window_,
                                   YWindow& y_window,
                                   MeanWindow& mean_window,
                                   InvStdWindow& inv_std_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        auto beta_window = make_tile_window(
            beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());

        // Problem::BlockShape
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));

        // total number of count assume current iter have no pad(only last iter has pad)
        constexpr index_t count_per_iter =
            Problem::BlockShape::Repeat_N * Problem::BlockShape::Vector_N;
        const index_t last_iter_n = row_size - (num_n_tile_iteration - 1) * Block_N;

        int cur_count = 0;
        int max_count =
            (num_n_tile_iteration - 1) * count_per_iter +
            block_tile_welford_calculate_max_count<typename Problem::BlockShape>(last_iter_n);
        auto block_welford      = Policy::template GetBlockWelford<Problem>();
        auto block_welford_sync = Policy::template GetBlockWelfordSync<Problem>();
        auto block_welford_cross_warp_sync =
            Policy::template GetBlockWelfordCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto mean         = block_welford.template MakeMeanVarBlockTile<XTensorType>();
        auto var          = block_welford.template MakeMeanVarBlockTile<XTensorType>();

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_welford(x, mean, var, cur_count, max_count);
            move_tile_window(x_window, {0, Block_N});
        }

        block_welford_sync(mean, var, cur_count);
        block_welford_cross_warp_sync(mean, var, cur_count, smem);
        block_tile_welford_post_scale_var(var, cur_count);

        // compute inv-std
        auto inv_std = tile_elementwise_in(
            [&](const auto& v_) {
                return type_convert<ComputeDataType>(1.0f) / (sqrt(v_ + epsilon));
            },
            var);

        if constexpr(kSaveMean)
            store_tile(mean_window, cast_tile<MeanDataType>(mean));
        if constexpr(kSaveInvStd)
            store_tile(inv_std_window, cast_tile<InvStdDataType>(inv_std));

        // reverse read x to reuse cache
        ck_tile::index_t stride_to_right_most_window =
            row_size % Block_N == 0 ? row_size - Block_N : row_size - row_size % Block_N;

        move_tile_window(x_window, {0, -Block_N});
        move_tile_window(gamma_window, {stride_to_right_most_window});
        move_tile_window(beta_window, {stride_to_right_most_window});
        move_tile_window(y_window, {0, stride_to_right_most_window});

        // layernorm computation
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            // load gamma/beta (TODO: support no gamma/beta?)
            const auto gamma = load_tile(gamma_window);
            const auto beta  = load_tile(beta_window);

            auto y = make_static_distributed_tensor<YDataType>(x.get_tile_distribution());

            sweep_tile(y, [&, mean_ = mean](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);

                const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);
                const auto beta_  = type_convert<ComputeDataType>(beta[j_idx]);

                const auto x_ = type_convert<ComputeDataType>(x[idx]);
                auto y_       = (x_ - mean_[i_idx]) * inv_std[i_idx] * gamma_ + beta_;

                y(idx) = type_convert<YDataType>(y_);
            });

            store_tile(y_window, y);

            move_tile_window(x_window, {0, -Block_N});
            move_tile_window(gamma_window, {-Block_N});
            move_tile_window(beta_window, {-Block_N});
            move_tile_window(y_window, {0, -Block_N});
        }
    }
};
} // namespace ck_tile
