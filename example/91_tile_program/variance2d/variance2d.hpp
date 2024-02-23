// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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

// Welford algorithm
template <typename XDataType,
          typename ComputeDataType,
          typename MeanDataType,
          typename VarDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock>
struct Variance2d
{
    struct Kargs
    {
        const void* p_x;
        void* p_mean;
        void* p_var;

        ck::index_t M;
        ck::index_t N;
    };

    __host__ static constexpr Kargs
    MakeKargs(const void* p_x, void* p_mean, void* p_var, ck::index_t M, ck::index_t N)
    {
        return Kargs{p_x, p_mean, p_var, M, N};
    }

    __host__ static constexpr auto GridSize(ck::index_t M) { return M / kMPerBlock; }

    __host__ static constexpr auto BlockSize() { return kBlockSize; }

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

    template <typename TileWindowWithStaticLengths, typename StaticDistributedTensor>
    __device__ static void StoreReduceTile1D(TileWindowWithStaticLengths& tile_window,
                                             const StaticDistributedTensor& dstr_tensor)
    {
        // TODO: Parse 32 from distribution
        if(ck::get_lane_id() % 32 == 0)
            store_tile(tile_window, dstr_tensor);
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::thread;
        using namespace ck::tile_program::warp;

        const XDataType* p_x = static_cast<const XDataType*>(kargs.p_x);
        MeanDataType* p_mean = static_cast<MeanDataType*>(kargs.p_mean);
        VarDataType* p_var   = static_cast<VarDataType*>(kargs.p_var);
        ck::index_t M        = kargs.M;
        ck::index_t N        = kargs.N;

        const auto x_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        constexpr auto xDstr = MakeXBlockTileDistribution();

        auto x_block_window = make_tile_window(
            x_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0}, xDstr);

        // TODO: padding - handle max_count if N % kNPerBlock != 0
        constexpr auto NPerThread = GetNPerThread(xDstr);
        ThreadWelford<ComputeDataType, XDataType> thread_welford{
            type_convert<int>(NPerThread * N / kNPerBlock)};

        auto mean_var_compute_block_tensor_tuple =
            decltype(thread_welford(load_tile(x_block_window))){};
        auto mean_compute_block_tensor = mean_var_compute_block_tensor_tuple.At(Number<0>{});
        auto var_compute_block_tensor  = mean_var_compute_block_tensor_tuple.At(Number<1>{});

        clear_tile(mean_compute_block_tensor);
        clear_tile(var_compute_block_tensor);

        index_t iN = 0;
        do
        {
            const auto x_block_tensor = load_tile(x_block_window);

            thread_welford(x_block_tensor, mean_compute_block_tensor, var_compute_block_tensor);
            move_tile_window(x_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // TODO: support cross warp reduction
        WarpMergeWelford<ComputeDataType, false>{}(
            mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

        // mean
        const auto mean_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            p_mean, make_tuple(M), Number<32>{});

        auto mean_block_window = make_tile_window(mean_m, make_tuple(Number<kMPerBlock>{}), {iM});

        StoreReduceTile1D(mean_block_window, cast_tile<MeanDataType>(mean_compute_block_tensor));

        // variance
        const auto var_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            p_var, make_tuple(M), Number<32>{});

        auto var_block_window = make_tile_window(var_m, make_tuple(Number<kMPerBlock>{}), {iM});

        StoreReduceTile1D(var_block_window, cast_tile<VarDataType>(var_compute_block_tensor));
    }
};
