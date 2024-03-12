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
template <typename Problem_>
struct Layernorm2dFwd
{
    using Problem = ck::remove_cvref_t<Problem_>;

    using XDataType       = ck::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kHasGamma   = !ck::is_same_v<GammaDataType, ck::null_type>;
    static constexpr bool kHasBeta    = !ck::is_same_v<BetaDataType, ck::null_type>;
    static constexpr bool kSaveMean   = !ck::is_same_v<MeanDataType, ck::null_type>;
    static constexpr bool kSaveInvStd = !ck::is_same_v<InvStdDataType, ck::null_type>;

    static constexpr ck::index_t kMPerBlock = Problem::BlockShape::kMPerBlock;
    static constexpr ck::index_t kNPerBlock = Problem::BlockShape::kNPerBlock;

    static constexpr ck::index_t kNThreadPerWarp = Problem::BlockShape::kNThreadPerWarp;

    struct Kargs
    {
        const void* p_x;
        const void* p_gamma;
        const void* p_beta;

        void* p_y;
        void* p_mean;
        void* p_invStd;

        float epsilon;

        ck::index_t M;
        ck::index_t N;
    };

    __host__ static constexpr Kargs MakeKargs(const void* p_x,
                                              const void* p_gamma,
                                              const void* p_beta,
                                              void* p_y,
                                              void* p_mean,
                                              void* p_invStd,
                                              float epsilon,
                                              ck::index_t M,
                                              ck::index_t N)
    {
        return Kargs{p_x, p_gamma, p_beta, p_y, p_mean, p_invStd, epsilon, M, N};
    }

    __host__ static constexpr auto GridSize(ck::index_t M) { return M / kMPerBlock; }

    __host__ static constexpr auto BlockSize() { return Problem::BlockShape::kBlockSize; }

    __device__ static constexpr auto MakeXBlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        constexpr ck::index_t kMPerThread = Problem::BlockShape::kMPerThread;
        constexpr ck::index_t kNPerThread = Problem::BlockShape::kNPerThread;

        constexpr ck::index_t kMThreadPerWarp = Problem::BlockShape::kMThreadPerWarp;

        constexpr index_t kMWarpPerBlock = Problem::BlockShape::kMWarpPerBlock;
        constexpr index_t kNWarpPerBlock = Problem::BlockShape::kNWarpPerBlock;

        // 4x1 wave
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<
                Sequence<>,
                Tuple<Sequence<kMWarpPerBlock, kMThreadPerWarp, kMPerThread>,
                      Sequence<kNWarpPerBlock, kNThreadPerWarp, kNPerThread>>,
                Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                Tuple<Sequence<0, 0>, Sequence<1, 1>>,
                Sequence<1, 2>,
                Sequence<2, 2>>{});
    }

    template <typename Dstr>
    __device__ static constexpr auto GetNPerThread(Dstr)
    {
        constexpr auto nDstrSpan = Dstr::GetDistributedSpans().template At<1>();

        using Lengths = decltype(nDstrSpan.impl_);

        ck::index_t ret = 1;

        ck::static_for<0, Lengths::Size(), 1>{}(
            [&](auto idx) { ret *= Lengths::template At(idx); });

        return ret;
    }

    template <typename DistributedTensor>
    __device__ static auto InvSqrt(const DistributedTensor& in_dstr_tensor,
                                   const ComputeDataType epsilon)
    {
        // TODO: Investigate fast inverse square root algorithm with epsilon
        using namespace ck;
        constexpr auto spans = DistributedTensor::GetDistributedSpans();

        DistributedTensor out_dstr_tensor;

        sweep_tile_span(spans[Number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            out_dstr_tensor(i_idx) =
                type_convert<ComputeDataType>(1.0f) / math::sqrt(in_dstr_tensor[i_idx] + epsilon);
        });

        return out_dstr_tensor;
    }

    template <bool Cond = (kHasGamma && kHasBeta)>
    __device__ ck::enable_if_t<Cond> TwoPassLayernorm2dFwd(const XDataType* p_x,
                                                           const GammaDataType* p_gamma,
                                                           const BetaDataType* p_beta,
                                                           YDataType* p_y,
                                                           MeanDataType* p_mean,
                                                           InvStdDataType* p_invStd,
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

        index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(N / kNPerBlock);

        // TODO: padding - handle max_count if N % kNPerBlock != 0
        constexpr auto NPerThread = GetNPerThread(xDstr);
        ThreadWelford<ComputeDataType, XDataType> thread_welford{
            type_convert<int>(NPerThread * N / kNPerBlock)};

        using XTensorType = decltype(load_tile(x_block_window));
        auto mean_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();
        auto var_compute_block_tensor =
            thread_welford.template MakeInitialMeanVarDistributedTensor<XTensorType>();

        clear_tile(mean_compute_block_tensor);
        clear_tile(var_compute_block_tensor);

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_block_tensor = load_tile(x_block_window);

            thread_welford(x_block_tensor, mean_compute_block_tensor, var_compute_block_tensor);
            move_tile_window(x_block_window, {0, kNPerBlock});
        }

        // TODO: support cross warp Welford
        WarpMergeWelford<ComputeDataType, true>{}(
            mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

        auto inv_std_compute_block_tensor = InvSqrt(var_compute_block_tensor, epsilon);

        if constexpr(kSaveMean)
        {
            const auto mean_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                p_mean, make_tuple(M), Number<32>{});

            auto mean_block_window =
                make_tile_window(mean_m, make_tuple(Number<kMPerBlock>{}), {iM});

            store_tile(mean_block_window, cast_tile<MeanDataType>(mean_compute_block_tensor));
        }
        if constexpr(kSaveInvStd)
        {
            const auto inv_std_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
                p_invStd, make_tuple(M), Number<32>{});

            auto inv_std_block_window =
                make_tile_window(inv_std_m, make_tuple(Number<kMPerBlock>{}), {iM});

            store_tile(inv_std_block_window, cast_tile<MeanDataType>(inv_std_compute_block_tensor));
        }

        // TODO: Extract normalize pipeline
        const auto y_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        auto y_block_window = make_tile_window(
            y_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0});

        constexpr auto gammaDstr = MakeXBlockTileDistribution();
        constexpr auto betaDstr  = gammaDstr;

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
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x_block_tensor     = load_tile(x_block_window);
            const auto gamma_block_tensor = load_tile(gamma_block_window);
            const auto beta_block_tensor  = load_tile(beta_block_window);

            constexpr auto x_spans = decltype(x_block_tensor)::GetDistributedSpans();

            auto y_block_tensor =
                make_static_distributed_tensor<YDataType>(x_block_tensor.GetTileDistribution());

            sweep_tile_span(x_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto mean    = mean_compute_block_tensor[i_idx];
                const auto inv_std = inv_std_compute_block_tensor[i_idx];

                sweep_tile_span(x_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    const auto x     = type_convert<ComputeDataType>(x_block_tensor[i_j_idx]);
                    const auto gamma = type_convert<ComputeDataType>(gamma_block_tensor[i_j_idx]);
                    const auto beta  = type_convert<ComputeDataType>(beta_block_tensor[i_j_idx]);
                    auto y           = (x - mean) * inv_std * gamma + beta;

                    y_block_tensor(i_j_idx) = type_convert<YDataType>(y);
                });
            });

            store_tile(y_block_window, y_block_tensor);

            move_tile_window(x_block_window, {0, -kNPerBlock});
            move_tile_window(gamma_block_window, {0, -kNPerBlock});
            move_tile_window(beta_block_window, {0, -kNPerBlock});
            move_tile_window(y_block_window, {0, -kNPerBlock});
        }
    }

    __device__ void operator()(Kargs kargs) const
    {
        TwoPassLayernorm2dFwd(static_cast<const XDataType*>(kargs.p_x),
                              static_cast<const GammaDataType*>(kargs.p_gamma),
                              static_cast<const BetaDataType*>(kargs.p_beta),
                              static_cast<YDataType*>(kargs.p_y),
                              static_cast<MeanDataType*>(kargs.p_mean),
                              static_cast<InvStdDataType*>(kargs.p_invStd),
                              static_cast<const ComputeDataType>(kargs.epsilon),
                              kargs.M,
                              kargs.N);
    }
};
