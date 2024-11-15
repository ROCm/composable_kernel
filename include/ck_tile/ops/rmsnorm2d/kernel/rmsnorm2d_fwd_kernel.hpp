// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// host side args
struct Rmsnorm2dFwdHostArgs
{
    const void* p_x;     // [m ,n], input, fp16/bf16
    const void* p_gamma; // [1, n], gamma, prec same as input

    void* p_y;      // [m, n], output, fp16/bf16
    void* p_invRms; // [m, 1], output inv-rms, prec same as input, nullptr if not used

    float epsilon;

    index_t m;
    index_t n;
    index_t stride; // row_stride
};

// TODO: Extract some type to wrapper class
template <typename Pipeline_>
struct Rmsnorm2dFwd
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType       = remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = remove_cvref_t<typename Problem::YDataType>;
    using InvRmsDataType  = remove_cvref_t<typename Problem::InvRmsDataType>;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, null_type>;
    static constexpr bool kSaveInvRms = Problem::kSaveInvRms;

    static constexpr index_t Block_M = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N = Problem::BlockShape::Block_N;
    static constexpr bool kPadM      = false; // always no need to pad along M
    static constexpr bool kPadN      = Problem::kPadN;
    static constexpr bool kTwoPass   = Problem::kTwoPass;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;
    static constexpr index_t Vector_N        = Problem::BlockShape::Vector_N;
    static constexpr index_t Repeat_N        = Problem::BlockShape::Repeat_N;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct Kargs
    {
        const void* p_x;
        const void* p_gamma;

        void* p_y;
        void* p_invRms;

        float epsilon;

        index_t m;
        index_t n;
        index_t stride; // row_stride
    };
    using Hargs = Rmsnorm2dFwdHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_gamma,
                     hargs.p_y,
                     hargs.p_invRms,
                     hargs.epsilon,
                     hargs.m,
                     hargs.n,
                     hargs.stride};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return dim3(integer_divide_ceil(hargs.m, Block_M));
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
            if (kSaveInvRms) n += "_rms";
            if (kTwoPass) n += "_2p";
            return n; }();

        #define _SS_  std::string
        #define _TS_  std::to_string
        return _SS_("rmsnorm2d_fwd_") + _SS_(t2s<XDataType>::name) + "_" +
             _TS_(S_::Block_M) + "x" + _TS_(S_::Block_N) + "_" + _TS_(S_::WarpPerBlock_M) + "x" + _TS_(S_::WarpPerBlock_N) + "_" +
             _TS_(S_::Warp_M) + "x" + _TS_(S_::Warp_N) + "_" + _TS_(S_::Vector_M) + "x" + _TS_(S_::Vector_N) + "_" +
             _SS_(Pipeline::name) + surfix;
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto iM = get_block_id() * Block_M;

        const auto x_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        const auto gamma_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const GammaDataType*>(kargs.p_gamma),
                make_tuple(kargs.n),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<kPadN>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
        }();

        auto y_window = [&]() {
            auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YDataType*>(kargs.p_y),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        auto inv_rms_window = [&]() {
            if constexpr(kSaveInvRms)
            {
                const auto inv_rms_m = [&]() {
                    const auto inv_rms_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<InvRmsDataType*>(kargs.p_invRms),
                            make_tuple(kargs.m),
                            number<1>{});

                    return pad_tensor_view(
                        inv_rms_dram_naive, make_tuple(number<Block_M>{}), sequence<kPadM>{});
                }();
                return make_tile_window(inv_rms_m, make_tuple(number<Block_M>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}));
        }();

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(x_window,
                   gamma_window,
                   y_window,
                   inv_rms_window,
                   static_cast<const ComputeDataType>(kargs.epsilon),
                   kargs.n,
                   smem);
    }
};

} // namespace ck_tile
