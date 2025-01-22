// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

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

    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveInvRms = Problem::Traits::kSaveInvRms;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockRmsnorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::Traits::kPadN;
    static constexpr bool kFastFDiv          = Problem::Traits::kFastFDiv;
    static constexpr bool kWelford           = Problem::Traits::kWelford;
    static constexpr auto kFusedAdd          = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant        = Problem::Traits::kFusedQuant;

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
              typename XResidualWindow,
              typename GammaWindow,
              typename YWindow,
              typename YResidualWindow,
              typename InvRmsWindow,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XResidualWindow& x_residual_window_,
                                   const GammaWindow& gamma_window_,
                                   YWindow& y_window,
                                   const YResidualWindow& y_residual_window_,
                                   InvRmsWindow& inv_rms_window,
                                   const SmoothScaleWindow& /*sm_scale_window_*/,
                                   YScaleWindow& /*y_scale_window*/,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem,
                                   Epilogue) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());
        auto x_residual_window = make_tile_window(
            x_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto y_residual_window = make_tile_window(
            y_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());

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
        auto block_norm_reduce      = Policy::template GetBlockNormReduce<Problem>();
        auto block_norm_reduce_sync = Policy::template GetBlockNormReduceSync<Problem>();
        auto block_norm_reduce_cross_warp_sync =
            Policy::template GetBlockNormReduceCrossWarpSync<Problem>();

        using XTensorType = decltype(cast_tile<ComputeDataType>(load_tile(x_window)));
        auto square_mean  = block_norm_reduce.template MakeMeanVarBlockTile<XTensorType>();
        clear_tile(square_mean);

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            auto x      = load_tile(x_window);
            auto x_resi = load_tile(x_residual_window);

            move_tile_window(x_window, {0, Block_N});
            move_tile_window(x_residual_window, {0, Block_N});

            auto acc = cast_tile<ComputeDataType>(x);

            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD ||
                         kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                sweep_tile(x_resi, [&](auto idx) {
                    // compute x = x_resi + x
                    acc(idx) = type_convert<ComputeDataType>(x_resi(idx)) + acc(idx);
                });
                if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
                {
                    store_tile(y_residual_window, cast_tile<YResidualDataType>(acc));
                    move_tile_window(y_residual_window, {0, Block_N});
                }
            }

            // Calculate square here because block norm reduce only supports naive mean.
            sweep_tile(acc, [&](auto idx) { acc(idx) *= acc(idx); });

            block_norm_reduce(acc, square_mean, cur_count, max_count);
        }

        block_norm_reduce_sync(square_mean, cur_count);
        block_norm_reduce_cross_warp_sync(square_mean, cur_count, smem);

        // compute inv-rms
        auto inv_rms = tile_elementwise_in(
            [&](const auto& v_) {
                if(kFastFDiv && std::is_same_v<ComputeDataType, float>)
                {
                    return type_convert<ComputeDataType>(1.0f) *
                           __builtin_amdgcn_rcpf(sqrt(v_ + epsilon));
                }
                else
                {
                    return type_convert<ComputeDataType>(1.0f) / (sqrt(v_ + epsilon));
                }
            },
            square_mean);

        if constexpr(kSaveInvRms)
            store_tile(inv_rms_window, cast_tile<InvRmsDataType>(inv_rms));

        // reverse read x to reuse cache
        ck_tile::index_t stride_to_right_most_window =
            row_size % Block_N == 0 ? row_size - Block_N : row_size - row_size % Block_N;

        move_tile_window(x_window, {0, -Block_N});
        move_tile_window(x_residual_window, {0, -Block_N});
        move_tile_window(gamma_window, {stride_to_right_most_window});
        move_tile_window(y_window, {0, stride_to_right_most_window});

        // rmsnorm computation
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            auto x      = load_tile(x_window);
            auto x_resi = load_tile(x_residual_window);
            auto acc    = cast_tile<ComputeDataType>(x);

            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE ||
                         kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD)
            {
                sweep_tile(x_resi, [&](auto idx) {
                    // compute x = x_resi + x
                    acc(idx) = type_convert<ComputeDataType>(x_resi(idx)) + acc(idx);
                });
            }

            // load gamma (TODO: support no gamma?)
            const auto gamma = load_tile(gamma_window);

            // rmsnorm computation
            auto rmsn = make_static_distributed_tensor<ComputeDataType>(x.get_tile_distribution());
            sweep_tile(rmsn, [&, inv_rms_ = inv_rms](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);

                const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);

                auto rmsn_ = acc(idx) * inv_rms_[i_idx] * gamma_;

                rmsn(idx) = rmsn_;
            });

            static_assert(kFusedQuant == Rmsnorm2dFusedQuantEnum::NO_SWEEP);
            Epilogue{}(y_window, rmsn);

            move_tile_window(x_window, {0, -Block_N});
            move_tile_window(x_residual_window, {0, -Block_N});
            move_tile_window(gamma_window, {-Block_N});
            move_tile_window(y_window, {0, -Block_N});
        }
    }
};
} // namespace ck_tile
