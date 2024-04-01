// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck/utility/common_header.hpp>
#include <ck/tile_program/tile/store_tile.hpp>
#include <ck/tile_program/tile/tile_elementwise.hpp>

template <typename OaccDataType_, typename ODataType_, bool kPadSeqLenQ_, bool kPadHeadDimV_>
struct FmhaFwdEpilogueProblem
{
    using OaccDataType                 = ck::remove_cvref_t<OaccDataType_>;
    using ODataType                    = ck::remove_cvref_t<ODataType_>;
    static constexpr bool kPadSeqLenQ  = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV = kPadHeadDimV_;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaFwdEpilogue
{
    using Problem                      = ck::remove_cvref_t<Problem_>;
    using OaccDataType                 = ck::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType                    = ck::remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename ODramWindowTmp, typename OAccTile>
    __device__ auto operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile)
    {
        using namespace ck;
        using namespace ck::tile_program;

        // TODO: this is ugly
        if constexpr(kPadSeqLenQ || kPadHeadDimV)
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
