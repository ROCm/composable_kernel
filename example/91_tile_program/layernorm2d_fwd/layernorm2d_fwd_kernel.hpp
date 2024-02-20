// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/thread_tile/thread_welford.hpp"
#include "ck/tile_program/warp_tile/warp_welford.hpp"
#include "ck/utility/functional2.hpp"

// TODO: Extract some type to wrapper class
template <typename Layernorm2dFwdPipeline_>
struct Layernorm2dFwd
{
    using Layernorm2dFwdPipeline = ck::remove_cvref_t<Layernorm2dFwdPipeline_>;

    using XDataType       = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::XDataType>;
    using GammaDataType   = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::GammaDataType>;
    using BetaDataType    = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::BetaDataType>;
    using ComputeDataType = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::ComputeDataType>;
    using YDataType       = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::YDataType>;
    using MeanDataType    = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::MeanDataType>;
    using InvStdDataType  = ck::remove_cvref_t<typename Layernorm2dFwdPipeline::InvStdDataType>;

    static constexpr bool HasGamma       = Layernorm2dFwdPipeline::Traits::HasGamma;
    static constexpr bool HasBeta        = Layernorm2dFwdPipeline::Traits::HasBeta;
    static constexpr bool SaveMeanInvStd = Layernorm2dFwdPipeline::Traits::SaveMeanInvStd;

    static constexpr ck::index_t kBlockSize = Layernorm2dFwdPipeline::kBlockSize;
    static constexpr ck::index_t kMPerBlock = Layernorm2dFwdPipeline::BlockLayernorm2dFwdShape::kM;
    static constexpr ck::index_t kNPerBlock = Layernorm2dFwdPipeline::BlockLayernorm2dFwdShape::kN;

    __device__ static constexpr auto MakeXBlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        // 4x1 wave
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<1, 4, 4, 2, 4>, Sequence<4, 1, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 2, 4>>{});
    }

    template <class Dstr>
    __device__ static constexpr auto GetVariance2dNPerThread(Dstr)
    {
        constexpr auto nDstrSpan = Dstr::GetDistributedSpans().template At<1>();

        using Lengths = decltype(nDstrSpan.impl_);

        ck::index_t ret = 1;

        ck::static_for<0, Lengths::Size(), 1>{}(
            [&](auto idx) { ret *= Lengths::template At(idx); });

        return ret;
    }

    template <typename ck::enable_if<HasGamma == true, bool>::type = false,
              typename ck::enable_if<HasBeta == true, bool>::type  = false>
    __device__ void TwoPassLayernorm2dFwd(const XDataType* p_x,
                                          const GammaDataType* p_gamma,
                                          const BetaDataType* p_beta,
                                          YDataType* p_y,
                                          MeanDataType* /*p_mean*/,
                                          InvStdDataType* /*p_invStd*/,
                                          const ComputeDataType epsilon,
                                          ck::index_t M,
                                          ck::index_t N) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::thread;
        using namespace ck::tile_program::warp;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const auto x_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto gamma_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_gamma, make_tuple(M, N), make_tuple(0, 1), Number<32>{}, Number<1>{});

        const auto beta_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_beta, make_tuple(M, N), make_tuple(0, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        constexpr auto xDstr = MakeXBlockTileDistribution();

        auto x_block_window = make_tile_window(
            x_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0}, xDstr);

        // TODO: padding - handle max_count if N % kNPerBlock != 0
        constexpr auto NPerThread = GetVariance2dNPerThread(xDstr);
        ThreadWelford<ComputeDataType, XDataType> thread_welford{NPerThread * N / kNPerBlock};

        auto mean_var_compute_block_tensor_tuple =
            decltype(thread_welford(load_tile(x_block_window))){};
        auto mean_compute_block_tensor = mean_var_compute_block_tensor_tuple.At(Number<0>{});
        auto var_compute_block_tensor  = mean_var_compute_block_tensor_tuple.At(Number<1>{});

        // init Mean & Var tile
        tile_elementwise_inout(
            [&](auto& mean, auto& var) { var = mean = type_convert<ComputeDataType>(0); },
            mean_compute_block_tensor,
            var_compute_block_tensor);

        index_t iN = 0;
        do
        {
            const auto x_block_tensor = load_tile(x_block_window);

            thread_welford(mean_compute_block_tensor, var_compute_block_tensor, x_block_tensor);
            move_tile_window(x_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // TODO: support cross warp Welford
        WarpMergeWelford<ComputeDataType, true>{}(
            mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

        // TODO: Extract normalize pipeline
        const auto y_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        auto y_block_window = make_tile_window(
            y_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0});

        constexpr auto gammaDstr = MakeXBlockTileDistribution();
        constexpr auto betaDstr  = MakeXBlockTileDistribution();

        auto gamma_block_window = make_tile_window(
            gamma_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0}, gammaDstr);

        auto beta_block_window = make_tile_window(
            beta_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0}, betaDstr);

        // reverse read x to reuse cache
        ck::index_t window_tail = N - kNPerBlock;

        move_tile_window(x_block_window, {0, -kNPerBlock});
        move_tile_window(gamma_block_window, {0, window_tail});
        move_tile_window(beta_block_window, {0, window_tail});
        move_tile_window(y_block_window, {0, window_tail});

        // Normalization
        do
        {
            const auto x_block_tensor     = load_tile(x_block_window);
            const auto gamma_block_tensor = load_tile(gamma_block_window);
            const auto beta_block_tensor  = load_tile(beta_block_window);

            constexpr auto x_spans = decltype(x_block_tensor)::GetDistributedSpans();

            auto y_block_tensor =
                make_static_distributed_tensor<YDataType>(x_block_tensor.GetTileDistribution());

            sweep_tile_span(x_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                sweep_tile_span(x_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    ComputeDataType x = type_convert<ComputeDataType>(x_block_tensor[i_j_idx]);
                    ComputeDataType gamma =
                        type_convert<ComputeDataType>(gamma_block_tensor[i_j_idx]);
                    ComputeDataType beta =
                        type_convert<ComputeDataType>(beta_block_tensor[i_j_idx]);
                    ComputeDataType mean = mean_compute_block_tensor[i_idx];
                    ComputeDataType var  = var_compute_block_tensor[i_idx];
                    ComputeDataType inv_std =
                        type_convert<ComputeDataType>(1.0f) / ck::math::sqrt(var + epsilon);
                    ComputeDataType y = (x - mean) * inv_std * gamma + beta;

                    y_block_tensor(i_j_idx) = type_convert<YDataType>(y);
                });
            });

            store_tile(y_block_window, y_block_tensor);

            move_tile_window(x_block_window, {0, -kNPerBlock});
            move_tile_window(gamma_block_window, {0, -kNPerBlock});
            move_tile_window(beta_block_window, {0, -kNPerBlock});
            move_tile_window(y_block_window, {0, -kNPerBlock});

            iN -= kNPerBlock;

        } while(iN > 0);

        if constexpr(SaveMeanInvStd)
        {
            // TODO
        }
    }

    __device__ void operator()(const XDataType* p_x,
                               const GammaDataType* p_gamma,
                               const BetaDataType* p_beta,
                               YDataType* p_y,
                               MeanDataType* p_mean,
                               InvStdDataType* p_invStd,
                               const ComputeDataType epsilon,
                               ck::index_t M,
                               ck::index_t N) const
    {
        TwoPassLayernorm2dFwd(p_x, p_gamma, p_beta, p_y, p_mean, p_invStd, epsilon, M, N);
    }
};
