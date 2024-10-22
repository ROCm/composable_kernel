// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dFwdPipelineDefaultPolicy>
struct Layernorm2dFwdPipelineOnePass
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
            return "bpr"; // block per row
        else
            return "wpr"; // warp per row
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
        const auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        const auto beta_window = make_tile_window(
            beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());

        const auto x  = load_tile(x_window);
        int cur_count = 0;
        int max_count =
            block_tile_welford_calculate_max_count<typename Problem::BlockShape>(row_size);
        auto block_welford      = Policy::template GetBlockWelford<Problem>();
        auto block_welford_sync = Policy::template GetBlockWelfordSync<Problem>();
        auto block_welford_cross_warp_sync =
            Policy::template GetBlockWelfordCrossWarpSync<Problem>();

        // load gamma/beta (TODO: support no gamma/beta?)
        const auto gamma = load_tile(gamma_window);
        const auto beta  = load_tile(beta_window);

        // compute welford each-thread->cross-lane->cross-warp
        auto [mean, var] = block_welford(x, cur_count, max_count);
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

        // layernorm computation
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
    }
};
} // namespace ck_tile
