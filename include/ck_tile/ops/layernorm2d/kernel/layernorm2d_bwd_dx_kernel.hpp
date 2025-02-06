// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// host side args
struct Layernorm2dBwdHostArgs
{
    const void* p_x;
    const void* p_dY;
    const void* p_gamma;
    const void* p_mean;
    const void* p_invStd;

    void* p_dGamma;
    void* p_dBeta;
    void* p_dX;

    //tmp
    void* p_dS;
    void* p_dB;

    index_t m;
    index_t n;
    index_t stride; // row_stride
};

// TODO: Extract some type to wrapper class
template <typename Pipeline_>
struct Layernorm2dBwdDX
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType       = remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr index_t Block_M = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N = Problem::BlockShape::Block_N;
    static constexpr index_t Vector_N = Problem::BlockShape::Vector_N;
    static constexpr bool kPadM      = false; // always no need to pad along M
    static constexpr bool kPadN      = Problem::kPadN;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct Kargs
    {
        const void* p_x;
        const void* p_dY;
        const void* p_gamma;
        const void* p_mean;
        const void* p_invStd;

        void* p_dGamma;
        void* p_dBeta;
        void* p_dX;

        //tmp
        void* p_dS;
        void* p_dB;

        index_t m;
        index_t n;
        index_t stride; // row_stride
    };
    using Hargs = Layernorm2dBwdHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_dY,
                     hargs.p_gamma,
                     hargs.p_mean,
                     hargs.p_invStd,
                     hargs.p_dGamma,
                     hargs.p_dBeta,
                     hargs.p_dX,

                     //tmp
                     hargs.p_dS,
                     hargs.p_dB,

                     hargs.m,
                     hargs.n,
                     hargs.stride};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return (hargs.m + Block_M - 1) / Block_M;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return Problem::BlockShape::BlockSize; }

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    // in byte
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        using S_ = typename Problem::BlockShape;
        auto surfix = [&] () {
            std::string n;
            if (kPadN) n += "_pn";
            return n; }();

        #define _SS_  std::string
        #define _TS_  std::to_string
        return _SS_("layernorm2d_bwd_") + _SS_(t2s<XDataType>::name) + "_" + 
             _TS_(S_::Block_M) + "x" + _TS_(S_::Block_N) + "_" + _TS_(S_::WarpPerBlock_M) + "x" + _TS_(S_::WarpPerBlock_N) + "_" +
             _TS_(S_::Warp_M) + "x" + _TS_(S_::Warp_N) + "_" + _TS_(S_::Vector_M) + "x" + _TS_(1) + "_" +
             _SS_(Pipeline::name) + surfix;
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto block_id = get_block_id();
        const auto iM = block_id * Block_M;

        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("dteng block shape---WarpPerBlock_M=%d, WarpPerBlock_N=%d, ThreadPerWarp_M=%d, ThreadPerWarp_N=%d, Vector_N=%d\n", static_cast<int>(Problem::BlockShape::WarpPerBlock_M), static_cast<int>(Problem::BlockShape::WarpPerBlock_N), static_cast<int>(Problem::BlockShape::ThreadPerWarp_M), static_cast<int>(Problem::BlockShape::ThreadPerWarp_N), static_cast<int>(Problem::BlockShape::Vector_N));
        // }
        
        const auto x_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            // NOTE: we don't do any pad in this kernel for loading, assume that inside kernel will
            // check the max count dynamically
            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        const auto dy_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const YDataType*>(kargs.p_dY),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            // NOTE: we don't do any pad in this kernel for loading, assume that inside kernel will
            // check the max count dynamically
            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();
                
        const auto gamma_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const MeanDataType*>(kargs.p_gamma),
                make_tuple(kargs.n),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<false>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
        }();

        const auto mean_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const MeanDataType*>(kargs.p_mean),
                make_tuple(kargs.m),
                make_tuple(1));

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {iM});
        }();

        const auto invstd_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const MeanDataType*>(kargs.p_invStd),
                make_tuple(kargs.m),
                make_tuple(1));

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {iM});
        }();

        auto dgamma_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<GammaDataType*>(kargs.p_dGamma),
                make_tuple(gridDim.x, kargs.n),
                make_tuple(kargs.n, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<1>{}, number<Block_N>{}), sequence<false, kPadN>{});

            return make_tile_window(tmp2_, make_tuple(number<1>{}, number<Block_N>{}), {block_id, 0});
        }();

        auto dbeta_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<BetaDataType*>(kargs.p_dBeta),
                make_tuple(gridDim.x, kargs.n),
                make_tuple(kargs.n, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<1>{}, number<Block_N>{}), sequence<false, kPadN>{});
            return make_tile_window(tmp2_, make_tuple(number<1>{}, number<Block_N>{}), {block_id, 0});
        }();

        auto dx_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<XDataType*>(kargs.p_dX),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        //tmp
        const auto ds_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<ComputeDataType*>(kargs.p_dS),
                make_tuple(kargs.m),
                make_tuple(1));

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {iM});
        }();

        const auto db_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<ComputeDataType*>(kargs.p_dB),
                make_tuple(kargs.m),
                make_tuple(1));

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {iM});
        }();

        // __shared__ char smem[GetSmemSize()];
        __shared__ char smem[0];

        Pipeline{}(x_window,
                   dy_window,
                   gamma_window,
                   mean_window,
                   invstd_window,
                   dgamma_window,
                   dbeta_window,
                   dx_window,

                   //tmp
                   ds_window,
                   db_window,

                   kargs.n,
                   smem);
    }
};

} // namespace ck_tile
