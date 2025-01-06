// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "rmsnorm2d_fwd.hpp"

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename XDataType_,
          typename YDataType_,
          typename XScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveInvRms_,
          bool kTwoPass_,
          ck_tile::index_t kFusedAdd_ = 0,
          ck_tile::index_t kFusedQuant_ = 0>
struct rmsnorm2d_fwd_traits_
{
    using XDataType = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType = ck_tile::remove_cvref_t<YDataType_>;
    using XScaleDataType = ck_tile::remove_cvref_t<XScaleDataType_>;
    using YScaleDataType = ck_tile::remove_cvref_t<YScaleDataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        }
        else
        {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool kPadN       = kPadN_;
    static constexpr bool kSaveInvRms = kSaveInvRms_;
    static constexpr bool kTwoPass    = kTwoPass_;
    static constexpr ck_tile::index_t kFusedAdd = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant = kFusedQuant_;
};

template <typename XDataType_,
          typename YDataType_,
          typename XScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveInvRms_,
          bool kTwoPass_,
          int  kFusedAdd_,
          int  kFusedQuant_>
using trait_ = rmsnorm2d_fwd_traits_<XDataType_,
                                     YDataType_,
                                     XScaleDataType_,
                                     YScaleDataType_,
                                     Repeat_M_,
                                     Repeat_N_,
                                     ThreadPerBlock_M_,
                                     ThreadPerBlock_N_,
                                     Vector_N_,
                                     kPadN_,
                                     kSaveInvRms_,
                                     kTwoPass_,
                                     kFusedAdd_,
                                     kFusedQuant_>;

template <typename data_type>
float rmsnorm2d_fwd_b16_(rmsnorm2d_fwd_traits /*t*/,
                         rmsnorm2d_fwd_args a,
                         const ck_tile::stream_config& s)
{
    float r = -1;
    // clang-format off
    //                                                                    rm  rn  tm   tn  vn  pd    rms     2p
    if(a.n <= 64) {
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  1,  4,  64, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 128) {
        if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  1,  4,  64, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  2,  4,  64, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 256) {
        if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  1,  4,  64, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  2,  4,  64, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  4,  4,  64, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 512) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  1,  4,  64, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  2,  4,  64, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  4,  4,  64, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  8,  4,  64, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 768) {
        if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  3,  4,  64, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  6,  4,  64, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 12,  4,  64, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 1024) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  1,  2,  128, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  2,  2,  128, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  4,  2,  128, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1,  4,  1,  256, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 1536) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 4,   64, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 2,  128, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 1,  256, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 6, 1,  256, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 2048) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 1, 1,  256, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 2, 1,  256, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 4, 1,  256, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 8, 1,  256, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 3072) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 1,  128, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 1,  256, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 6, 1,  256, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 3, 1, 1024, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n <= 4096) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 2, 1,  256, 8,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 4, 1,  256, 4,  true,  false, false, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 2, 1, 1024, 2,  true,  false, false, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 4, 1, 1024, 1,  true,  false, false, 0, 0>>(s, a);
    }
    else if(a.n > 4096) {
        if (a.n % 8 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 2, 1,  256, 8,  true,  false, true, 0, 0>>(s, a);
        else if (a.n % 4 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 4, 1,  256, 4,  true,  false, true, 0, 0>>(s, a);
        else if (a.n % 2 == 0)
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 2, 1, 1024, 2,  true,  false, true, 0, 0>>(s, a);
        else
            r = rmsnorm2d_fwd_<trait_<data_type, data_type, float, float,  1, 4, 1, 1024, 1,  true,  false, true, 0, 0>>(s, a);
    }
    return r;
    // clang-format on
}

float rmsnorm2d_fwd(rmsnorm2d_fwd_traits t, rmsnorm2d_fwd_args a, const ck_tile::stream_config& s)
{

    if ((t.prec_i.compare("fp16") == 0) && (t.prec_o.compare("fp16") == 0) && (t.fused_quant == 0))
    {
        return rmsnorm2d_fwd_b16_<ck_tile::fp16_t>(t, a, s);
    }
    else if ((t.prec_i.compare("bf16") == 0) && (t.prec_o.compare("fp16") == 0) && (t.fused_quant == 0))
    {
        return rmsnorm2d_fwd_b16_<ck_tile::bf16_t>(t, a, s);
    }
    else
        throw std::runtime_error("Without supported instances!");
}
