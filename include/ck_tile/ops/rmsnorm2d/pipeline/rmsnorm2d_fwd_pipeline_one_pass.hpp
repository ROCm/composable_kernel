// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Rmsnorm2dFwdPipelineDefaultPolicy>
struct Rmsnorm2dFwdPipelineOnePass
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

    static constexpr bool kHasGamma    = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveInvRms  = Problem::Traits::kSaveInvRms;
    static constexpr bool kSaveUnquant = Problem::Traits::kSaveUnquant;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockRmsnorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::Traits::kPadN;
    static constexpr auto kFusedAdd          = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant        = Problem::Traits::kFusedQuant;

    static constexpr auto Vector_N           = Problem::BlockShape::Vector_N;
    static constexpr auto Repeat_N           = Problem::BlockShape::Repeat_N;
    static constexpr auto Stride_N           = Problem::BlockShape::Block_N / Repeat_N;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_op"; // block per row
        else
            return "wpr_op"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_DEVICE static constexpr auto MakeSmoothInputScaleTileDistribution()
    {
        using S = Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
    }

    template <typename XWindow,
              typename XResidualWindow,
              typename GammaWindow,
              typename YWindow,
              typename YResidualWindow,
              typename InvRmsWindow,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename UnquantYWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XResidualWindow& x_residual_window_,
                                   const GammaWindow& gamma_window_,
                                   YWindow& y_window_,
                                   const YResidualWindow& y_residual_window_,
                                   InvRmsWindow& inv_rms_window,
                                   const SmoothScaleWindow& sm_scale_window_,
                                   YScaleWindow& y_scale_window_,
                                   UnquantYWindow& unquant_y_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem,
                                   Epilogue) const
    {
        auto x_window =
            make_tile_window(x_window_.get_bottom_tensor_view(),
                             x_window_.get_window_lengths(),
                             x_window_.get_window_origin(),
		                     Policy::template MakeXInnerBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());
        auto x_residual_window =
            make_tile_window(x_residual_window_.get_bottom_tensor_view(),
                             x_residual_window_.get_window_lengths(),
                             x_residual_window_.get_window_origin(),
		                     Policy::template MakeXInnerBlockTileDistribution<Problem>());
        auto y_residual_window =
            make_tile_window(y_residual_window_.get_bottom_tensor_view(),
                             y_residual_window_.get_window_lengths(),
                             y_residual_window_.get_window_origin(),
		                     Policy::template MakeXInnerBlockTileDistribution<Problem>());

        auto o_window =
            make_tile_window(y_window_.get_bottom_tensor_view(),
                             make_tuple(number<Problem::BlockShape::Block_M>{}, number<Problem::BlockShape::Block_N / Repeat_N>{}),
                             y_window_.get_window_origin(),
		                     Policy::template MakeXInnerBlockTileDistribution<Problem>());

        auto o_all_window =
            make_tile_window(y_window_.get_bottom_tensor_view(),
                             y_window_.get_window_lengths(),
                             y_window_.get_window_origin(),
		                     Policy::template MakeXBlockTileDistribution<Problem>());
        auto reduce_square_sum_func = ReduceOp::SquareAdd{};
        auto reduce_sum_func        = ReduceOp::Add{};
        auto block_reduce2d         = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync    = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using AccTensorType = decltype(cast_tile<ComputeDataType>(load_tile(x_window)));
        using AccResTensorType = decltype(load_tile(x_residual_window));

        AccTensorType x_warp_tensors[Repeat_N];
        AccTensorType o_warp_tensors[Repeat_N];

        auto square_sum = decltype(block_reduce2d(AccTensorType{},
                                                  reduce_square_sum_func.GetIdentityValue<ComputeDataType>(),
                                                  reduce_square_sum_func)){};
        clear_tile(square_sum);

        const auto sm_scale_window =
            make_tile_window(sm_scale_window_, MakeSmoothInputScaleTileDistribution());

        for (int repeat_n = 0; repeat_n < Repeat_N; ++repeat_n)
        {
            auto x = load_tile(x_window);
            x_window.move({0, Stride_N});

            auto x_resi = load_tile(x_residual_window);
            if constexpr(x_resi.is_valid())
                move_tile_window(x_residual_window, {0, Stride_N});

            // load gamma (TODO: support no gamma?)
            

            x_warp_tensors[repeat_n] = cast_tile<ComputeDataType>(x);

            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD ||
                         kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                sweep_tile(x_resi, [&](auto idx) {
                    // compute x = x_resi + x
                    x_warp_tensors[repeat_n](idx) = type_convert<ComputeDataType>(x_resi(idx)) + x_warp_tensors[repeat_n](idx);
                });
            }

            // compute mean square each-thread->cross-lane->cross-warp
            auto square_sum_local = block_reduce2d(x_warp_tensors[repeat_n],
                                        reduce_square_sum_func.GetIdentityValue<ComputeDataType>(),
                                        reduce_square_sum_func);

            ck_tile::sweep_tile(square_sum, [&](auto idx) {
                square_sum(idx) += square_sum_local[idx];
            });
        }

        const auto gamma = load_tile(gamma_window);

        block_reduce2d_sync(square_sum, reduce_sum_func);
        block_reduce2d_cross_warp_sync(square_sum, smem, reduce_sum_func);

        auto sm_scale = load_tile(sm_scale_window);

        // compute inv-rms
        auto inv_rms = tile_elementwise_in(
            [&](const auto& v_) { return rsqrtf(v_ / row_size + epsilon); }, square_sum);

        if constexpr(kSaveInvRms)
            store_tile(inv_rms_window, cast_tile<InvRmsDataType>(inv_rms));

        // rmsnorm computation
        auto rmsn = make_static_distributed_tensor<ComputeDataType>(Policy::template MakeXBlockTileDistribution<Problem>());

        static_for<0, Repeat_N, 1>{}([&](auto repeat_n)
        {
            sweep_tile(o_warp_tensors[0], [&, inv_rms_ = inv_rms](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);

                const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);

                auto rmsn_ = o_warp_tensors[repeat_n][idx] * inv_rms_[i_idx] * gamma_;

                if constexpr(sm_scale.is_valid())
                {
                    const auto xs_ = type_convert<ComputeDataType>(sm_scale[j_idx]);
                    o_warp_tensors[repeat_n](idx) = rmsn_ * xs_;
                }
            });
        });

        for (int repeat_n = 0; repeat_n < Repeat_N; ++repeat_n)
        {
			if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
			{
				store_tile(y_residual_window, cast_tile<YResidualDataType>(x_warp_tensors[repeat_n]));
				if constexpr(AccResTensorType::is_valid())
					move_tile_window(y_residual_window, {0, Stride_N});
			}
        }

        if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT)
        {
            if constexpr(kSaveUnquant)
            {
                Epilogue{}(
                    unquant_y_window, o_all_window, sm_scale_window_, y_scale_window_, o_warp_tensors, true, smem);
            }
            else
            {
                Epilogue{}(o_window, sm_scale_window_, y_scale_window_, o_warp_tensors, true, smem);
            }
        }
        else if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT)
        {
            if constexpr(kSaveUnquant)
            {
                Epilogue{}(unquant_y_window, o_all_window, y_scale_window_, rmsn, smem);
            }
            else
            {
                Epilogue{}(o_all_window, y_scale_window_, rmsn, smem);
            }
        }
        else
        {
            Epilogue{}(o_all_window, rmsn);
        }
    }
};
} // namespace ck_tile
