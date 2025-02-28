// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <string>

template <typename DataType>
struct LayerNormTypeConfig;

template <>
struct LayerNormTypeConfig<ck_tile::half_t>
{
    using XDataType       = ck_tile::half_t;
    using YDataType       = ck_tile::half_t;
    using GammaDataType   = ck_tile::half_t;
    using BetaDataType    = ck_tile::half_t;
    using MeanDataType    = ck_tile::half_t;
    using InvStdDataType  = ck_tile::half_t;
    using ComputeDataType = float;
};

template <>
struct LayerNormTypeConfig<ck_tile::bf16_t>
{
    using XDataType       = ck_tile::bf16_t;
    using YDataType       = ck_tile::bf16_t;
    using GammaDataType   = ck_tile::bf16_t;
    using BetaDataType    = ck_tile::bf16_t;
    using MeanDataType    = ck_tile::bf16_t;
    using InvStdDataType  = ck_tile::bf16_t;
    using ComputeDataType = float;
};

// runtime args
struct layernorm2d_bwd_args : public ck_tile::Layernorm2dBwdHostArgs
{
};

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_M_,          // vector size along M
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kTwoPass_,
          bool kCalData_>
struct layernorm2d_bwd_traits_
{
    using DataType = ck_tile::remove_cvref_t<DataType_>;

    static constexpr bool single_warp_first_dim = 1 ? ThreadPerBlock_N_ <= warpSize : ThreadPerBlock_M_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(1)
        {
            if constexpr(single_warp_first_dim)
            {
                static_assert(warpSize % ThreadPerBlock_N_ == 0);
                // return total_warps * (warpSize / ThreadPerBlock_N_);
                return total_warps;
            }
            else
            {
                // static_assert(warpSize % ThreadPerBlock_M_ == 0);
                return total_warps / (ThreadPerBlock_N_ / warpSize);
            }
        }
        else
        {
            if constexpr(single_warp_first_dim)
            {
                static_assert(warpSize % ThreadPerBlock_M_ == 0);
                return 1;
            }
            else
            {
                static_assert(ThreadPerBlock_M_ % warpSize == 0);
                return ThreadPerBlock_M_ / warpSize;
            }
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(1)
        {
            if constexpr(single_warp_first_dim)
            {
                static_assert(warpSize % ThreadPerBlock_N_ == 0);
                return 1;
            }
            else
            {
                static_assert(ThreadPerBlock_N_ % warpSize == 0);
                return ThreadPerBlock_N_ / warpSize;
            }
        }
        else
        {
            if constexpr(single_warp_first_dim)
            {
                static_assert(warpSize % ThreadPerBlock_M_ == 0);
                // return total_warps * (warpSize / ThreadPerBlock_M_);
                return total_warps;
            }
            else
            {
                // static_assert(warpSize % ThreadPerBlock_N_ == 0);
                return total_warps / (ThreadPerBlock_M_ / warpSize);
            }
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_ * Vector_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M * Vector_M_;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<Vector_M_, Vector_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kTwoPass        = kTwoPass_;
    static constexpr bool kCalData        = kCalData_;
};

template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_M_,         // vector size along M
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kTwoPass_,
          bool kCalData_>
using trait_ = layernorm2d_bwd_traits_<DataType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_M_,
                                       Vector_N_,
                                       kPadN_,
                                       kTwoPass_,
                                       kCalData_>;

template <typename Traits_>
float layernorm2d_bwd_(const ck_tile::stream_config& s, layernorm2d_bwd_args a);

// This is the public API, will be generated by script
struct layernorm2d_bwd_traits
{
    std::string DataType;
    bool CalData; // 0: weight grad, 1: data grad
};

template <typename DataType>
struct layernorm2d_bwd_b16_
{
    /* data */
    //using Trait = trait_<DataType,   1,  1,  1,  256,  1,  1,  true>;
    //using Trait = trait_<DataType,   1,  8,  64,  4,  1,  8,  true>;
    //using Trait = trait_<DataType,   1,  4,  1,  64,  1,  8,  true>;
    //using Trait = trait_<DataType,   1,  2,  4,  16, 1,  8,  true,  false,  true>;
    //using Trait = trait_<DataType,   1,  1,  64,  1,  1,  1,  true,  false,  false>;
    float operator() (layernorm2d_bwd_traits t,
                      layernorm2d_bwd_args a,
                      const ck_tile::stream_config& s) {
        // if (t.CalData)
        // {
        //     if (a.n <= 256)
        //         return layernorm2d_bwd_<trait_<DataType,  1,  2,  4,  16, 1,  8,  true,  false,  true>>(s, a);
        //     else
        //         return layernorm2d_bwd_<trait_<DataType,  1,  4,  2,  32, 1,  8,  true,  true,  true>>(s, a);
        // }
        // else
        // {
            // if (a.n <= 64)
            //     return layernorm2d_bwd_<trait_<DataType,  1,  1,  64,  1,  1,  1,  true,  false,  false>>(s, a);
            // else
                return layernorm2d_bwd_<trait_<DataType,  2,  1,  32,  16,  8,  2,  true,  false,  false>>(s, a);
                // return layernorm2d_bwd_<trait_<DataType,  1,  1,  8,  32,  1,  2,  true,  false,  false>>(s, a);
        // }
    }
};

// template <typename data_type>
// ck_tile::index_t layernorm2d_bwd_block_m() {
//     return layernorm2d_bwd_b16_<data_type>::Trait::Block_M;
// };

float layernorm2d_bwd(layernorm2d_bwd_traits, layernorm2d_bwd_args, const ck_tile::stream_config&);
