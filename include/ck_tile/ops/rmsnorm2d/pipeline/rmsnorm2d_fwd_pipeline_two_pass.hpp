// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Rmsnorm2dFwdPipelineDefaultPolicy>
struct Rmsnorm2dFwdPipelineTwoPass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using InvRmsDataType  = ck_tile::remove_cvref_t<typename Problem::InvRmsDataType>;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveInvRms = Problem::kSaveInvRms;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockRmsnorm2dFwdProblem::kPadM
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

    template <typename XWindow, typename GammaWindow, typename YWindow, typename InvRmsWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const GammaWindow& gamma_window_,
                                   YWindow& y_window,
                                   InvRmsWindow& inv_rms_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());

        // Problem::BlockShape
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));

        auto reduce_square_sum_func = ReduceOp::SquareAdd{};
        auto reduce_sum_func        = ReduceOp::Add{};
        auto block_reduce2d         = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync    = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto square_sum   = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(square_sum, reduce_square_sum_func.GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_reduce2d(x, square_sum, reduce_square_sum_func);
            move_tile_window(x_window, {0, Block_N});
        }

        block_reduce2d_sync(square_sum, reduce_sum_func);
        block_reduce2d_cross_warp_sync(square_sum, smem, reduce_sum_func);

        // compute inv-rms
        auto inv_rms = tile_elementwise_in(
            [&](const auto& v_) {
                return type_convert<ComputeDataType>(1.0f) / (sqrt(v_ / row_size + epsilon));
            },
            square_sum);

        if constexpr(kSaveInvRms)
            store_tile(inv_rms_window, cast_tile<InvRmsDataType>(inv_rms));

        // reverse read x to reuse cache
        ck_tile::index_t stride_to_right_most_window =
            row_size % Block_N == 0 ? row_size - Block_N : row_size - row_size % Block_N;

        move_tile_window(x_window, {0, -Block_N});
        move_tile_window(gamma_window, {stride_to_right_most_window});
        move_tile_window(y_window, {0, stride_to_right_most_window});

        // rmsnorm computation
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            // load gamma/beta (TODO: support no gamma/beta?)
            const auto gamma = load_tile(gamma_window);

            auto y = make_static_distributed_tensor<YDataType>(x.get_tile_distribution());

            sweep_tile(y, [&, inv_rms_ = inv_rms](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);

                const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);

                const auto x_ = type_convert<ComputeDataType>(x[idx]);
                auto y_       = x_ * inv_rms_[i_idx] * gamma_;

                y(idx) = type_convert<YDataType>(y_);
            });

            store_tile(y_window, y);

            move_tile_window(x_window, {0, -Block_N});
            move_tile_window(gamma_window, {-Block_N});
            move_tile_window(y_window, {0, -Block_N});
        }
    }
};
} // namespace ck_tile
