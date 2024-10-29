// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <bool kPadM_, bool kPadN_, bool UseRawStore_ = true, bool UseMax3_ = false>
struct DynamicQuantEpilogueTraits
{
    static constexpr bool kPadM       = kPadM_;
    static constexpr bool kPadN       = kPadN_;
    static constexpr bool UseRawStore = UseRawStore_;
    static constexpr bool UseMax3     = UseMax3_;
};

// this epilogue just store out a M*N matrix, row major
template <typename AccDataType_, typename YScaleDataType_, typename ODataType_, typename Traits_>
struct DynamicQuantEpilogueProblem
{
    using AccDataType    = remove_cvref_t<AccDataType_>;
    using YScaleDataType = remove_cvref_t<YScaleDataType_>;
    using ODataType      = remove_cvref_t<ODataType_>;
    using Traits         = remove_cvref_t<Traits_>;
};

template <typename Problem_, typename Policy_ = void>
struct DynamicQuantEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using YScaleDataType              = remove_cvref_t<typename Problem::YScaleDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::Traits::kPadM;
    static constexpr bool kPadN       = Problem::Traits::kPadN;
    static constexpr bool UseRawStore = Problem::Traits::UseRawStore;
    static constexpr bool UseMax3     = Problem::Traits::UseMax3;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename YScaleWindow, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   YScaleWindow& y_scale_window,
                                   const OAccTile& o_acc_tile)
    {
        // compute row max
        auto reduce_row_absmax = BlockReduce2D{o_acc_tile, type_convert<AccDataType>(0)};
        auto row_absmax        = [&]() {
            if constexpr(UseMax3 && std::is_same_v<AccDataType, float>)
            {
                const auto f_max = [](auto acc_, auto v_0_) { return max(acc_, abs(v_0_)); };
                // const auto f_max3 = [](auto acc_, auto v_0_, auto v_1_) {
                //     float rtn;
                //     asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                //                  : "=v"(rtn)
                //                  : "v"(acc_), "v"(v_0_), "v"(v_1_));
                //     return rtn;
                // };
                // return reduce_row_absmax(f_max3, f_max, sequence<1, 2>{});
                return reduce_row_absmax(f_max);
            }
            else
            {
                const auto f_max = [](auto acc_, auto v_0_) { return max(acc_, abs(v_0_)); };
                return reduce_row_absmax(f_max);
            }
        }();

        // here y_scale is Acc TYpe, need convert to YScale type later
        auto y_scale = tile_elementwise_in(
            [&](const auto& v_) {
                return v_ / type_convert<AccDataType>(numeric<ODataType>::max());
            },
            row_absmax);

        store_tile(y_scale_window, cast_tile<YScaleDataType>(y_scale));

        // TODO: this is ugly
        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            buffer_store_fence();
        }
        else
        {
            store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
        }
    }
};
} // namespace ck_tile
