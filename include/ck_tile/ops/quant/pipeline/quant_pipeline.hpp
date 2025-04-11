// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/quant/pipeline/quant_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {
template <typename Problem_, typename Policy_ = PerTensorQuantPipelineDefaultPolicy>
struct StaticPerTensorQuantPipeline
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType           = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ScaleDataType       = ck_tile::remove_cvref_t<typename Problem::ScaleDataType>;
    using ComputeDataType     = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using QXDataType          = ck_tile::remove_cvref_t<typename Problem::QXDataType>;


    static constexpr const char* name = []() {
        return "static_per_tensor_quant_op";
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename ScaleDataType,
              typename QXWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const ScaleDataType* scale,
                                   ck_tile::index_t row_size,
                                   QXWindow& qx_window,
                                   void*) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN){
            const auto x       = load_tile(x_window);
            const auto qx      = tile_elementwise_in(
                [&](const auto& a) {
                    return type_convert<QXDataType>(saturates<QXDataType>{}(type_convert<ComputeDataType>(a) / type_convert<ComputeDataType>(*scale)));
                },
                x);
            store_tile(qx_window, qx);
            move_tile_window(x_window, {0, Block_N});
            move_tile_window(qx_window, {0, Block_N});
        }
    }
};

template <typename Problem_, typename Policy_ = PerTensorQuantPipelineDefaultPolicy>
struct DynamicPerTensorQuantPipeline
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType           = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ScaleDataType       = ck_tile::remove_cvref_t<typename Problem::ScaleDataType>;
    using ComputeDataType     = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using QXDataType          = ck_tile::remove_cvref_t<typename Problem::QXDataType>;

    static constexpr bool UseMax3            = true;

    static constexpr const char* name = []() {
        return "dynamic_per_tensorquant_op";
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename ScaleDataType,
              typename QXWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   ScaleDataType* scale,
                                   ck_tile::index_t row_size,
                                   QXWindow& qx_window,
                                   void* smem) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto origin = x_window.get_window_origin();
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));

        auto reduce_absmax_func  = ReduceOp::AbsMax{};
        auto reduce_absmax3_func = [](auto acc_, auto v_0_, auto v_1_) {
            float rtn;
            asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                         : "=v"(rtn)
                         : "v"(acc_), "v"(v_0_), "v"(v_1_));
            return rtn;
        };
        auto reduce_max_func     = ReduceOp::Max{};
        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorType = decltype(cast_tile<ComputeDataType>(load_tile(x_window)));
        auto absmax       = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(absmax, reduce_absmax_func.GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN){
            const auto x       = load_tile(x_window);
            constexpr auto x_size_per_row =
                x.get_tile_distribution().get_ys_to_d_descriptor().get_lengths().at(number<1>{});
            if constexpr(UseMax3 && std::is_same_v<ComputeDataType, float> &&
                         x_size_per_row % 2 == 0)
                block_reduce2d(x, absmax, reduce_absmax3_func, sequence<1, 2>{});
            else
                block_reduce2d(x, absmax, reduce_absmax_func);
            move_tile_window(x_window, {0, Block_N});
        }
        block_reduce2d_sync(absmax, reduce_max_func);
        block_reduce2d_cross_warp_sync(absmax, smem, reduce_max_func);
        *scale = absmax.get_thread_buffer()[0] / ck_tile::numeric<QXDataType>::max();

        x_window.set_window_origin(origin);
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN){
            const auto x       = load_tile(x_window);
            const auto qx      = tile_elementwise_in(
                [&](const auto& a) {
                    return type_convert<QXDataType>(saturates<QXDataType>{}(type_convert<ComputeDataType>(a) / type_convert<ComputeDataType>(*scale)));
                },
                x);
            store_tile(qx_window, qx);
            move_tile_window(x_window, {0, Block_N});
            move_tile_window(qx_window, {0, Block_N});
        }
    }
};

template <typename Problem_, typename Policy_ = PerTensorQuantPipelineDefaultPolicy>
struct DynamicPerTokenQuantPipeline
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType           = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ScaleDataType       = ck_tile::remove_cvref_t<typename Problem::ScaleDataType>;
    using ComputeDataType     = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using QXDataType          = ck_tile::remove_cvref_t<typename Problem::QXDataType>;

    static constexpr bool UseMax3            = true;

    static constexpr const char* name = []() {
        return "dynamic_per_token_quant_op";
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 1;
    }

    template <typename XWindow,
              typename ScaleWindow,
              typename QXWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   ScaleWindow& scale_window,
                                   ck_tile::index_t row_size,
                                   QXWindow& qx_window,
                                   void*) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto origin = x_window.get_window_origin();
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));

        const auto reduce_absmax_func = ReduceOp::AbsMax{};
        auto reduce_absmax3_func = [](auto acc_, auto v_0_, auto v_1_) {
            float rtn;
            asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                         : "=v"(rtn)
                         : "v"(acc_), "v"(v_0_), "v"(v_1_));
            return rtn;
        };

        // auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();

        using XTensorType = decltype(cast_tile<ComputeDataType>(load_tile(x_window)));

        constexpr auto absmax_dstr =
            make_static_tile_distribution(ck_tile::detail::make_reduce_tile_distribution_encoding(
                XTensorType::get_tile_distribution().get_static_tile_distribution_encoding(),
                sequence<1>{}));

        auto absmax = make_static_distributed_tensor<ComputeDataType>(absmax_dstr);

        // auto absmax       = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(absmax, reduce_absmax_func.GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN){
            const auto x = cast_tile<ComputeDataType>(load_tile(x_window));
            constexpr auto x_size_per_row =
                x.get_tile_distribution().get_ys_to_d_descriptor().get_lengths().at(number<1>{});
            if constexpr(UseMax3 && std::is_same_v<ComputeDataType, float> &&
                         x_size_per_row % 2 == 0)
                block_tile_reduce(absmax, x, sequence<1>{}, reduce_absmax3_func);
            else
                block_tile_reduce(absmax, x, sequence<1>{}, reduce_absmax_func);
            move_tile_window(x_window, {0, Block_N});
        }

        auto scale_tile = tile_elementwise_in(
            [&](const auto& v_) {
                return v_ / type_convert<ComputeDataType>(numeric<QXDataType>::max());
            },
            absmax);
  
        store_tile(scale_window, cast_tile<ScaleDataType>(scale_tile));
        
        x_window.set_window_origin(origin);
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN){
            const auto x = cast_tile<ComputeDataType>(load_tile(x_window));
            auto qx = make_static_distributed_tensor<QXDataType>(x.get_tile_distribution());
            sweep_tile(x, [&](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                qx(idx) = type_convert<QXDataType>(saturates<QXDataType>{}(x[idx] / scale_tile[i_idx]));
            });
            store_tile(qx_window, qx);
            move_tile_window(x_window, {0, Block_N});
            move_tile_window(qx_window, {0, Block_N});
        }
    }
};
} // namespace ck_tile
