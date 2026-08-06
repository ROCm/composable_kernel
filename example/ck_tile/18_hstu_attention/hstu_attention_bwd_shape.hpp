// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
// Narrow fmha sub-header instead of the `ck_tile/ops/fmha.hpp` aggregate (which
// pulls in kernel/fmha_fwd_kernel.hpp and collides with the local
// hstu_attention_kernel_util.hpp). tensor_layout is what tile_fmha_shape.hpp
// needs for `tensor_layout::gemm::RowMajor`.
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"

// HSTU attention backward — per-MaxK tile-shape selector (M7b).
//
// The bwd dispatch is templated on MaxK but, pre-M7b, hardcoded the hd64 tile
// shape (so MaxK only flowed into the instance symbol, NOT the shape — adding a
// headdim axis without this would silently reuse the hd64 tile = wrong). This
// header makes the TileFmhaBwdShape a compile-time function of MaxK.
//
// Tile presets mirror the upstream FMHA bwd codegen for gfx950 / CDNA4 (non-
// trload fp16/bf16): example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py
//   KernelComponentFactoryGfx9::get_dq_dk_dv_tiles("fp16"/"bf16", "").
// FmhaBlockTile = sequence<bm0,bn0,bk0,bk1,bk2,bk3,bk4,bhdq,bhdv>.
// Gemm4 warp tile = sequence<wm0,wn0,min(wk0,bk4)>; for all four presets wk0=32
// and bk4=32, so min(wk0,bk4)=wk0 and the Gemm4 slot reuses WarpTile0 (= the
// existing hd64 dispatch's convention).
//
// ZERO-REGRESSION INVARIANT: HstuBwdShape<64>::Type MUST be the exact same type
// as the pre-M7b hardcoded FmhaBwdShape (same template args, same order), so the
// MaxK=64 path instantiates a byte-identical kernel. Verified by diffing the
// regenerated hd64 instances vs baseline + the 106-case suite.

template <ck_tile::index_t MaxK>
struct HstuBwdShape; // primary left undefined: only the four canonical hdims are valid.

// hd64 — byte-identical to the pre-M7b hardcoded preset in both dispatches.
template <>
struct HstuBwdShape<64>
{
    using FmhaBlockTile = ck_tile::sequence<32, 128, 64, 32, 64, 32, 32, 64, 64>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<1, 4, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

    // M7c: tile head-dim (bhdq/bhdv, square == MaxK) for the pad modulo predicate, and
    // kN0 (bn0, k-seqlen block) for the harness determ workspace sizing.
    static constexpr ck_tile::index_t kN0        = 128;
    static constexpr ck_tile::index_t kQKHeaddim = 64;
    static constexpr ck_tile::index_t kVHeaddim  = 64;

    using Type = ck_tile::TileFmhaBwdShape<FmhaBlockTile,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps2,
                                           WarpTile0,
                                           0 /* kMaxSeqLenQ: 0 = unlimited */>;
};

// hd96 — note BlockWarps2 = <2,2,1> (Gemm4 warp layout differs from 64/128/256).
template <>
struct HstuBwdShape<96>
{
    using FmhaBlockTile = ck_tile::sequence<32, 128, 96, 32, 96, 32, 32, 96, 96>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<2, 2, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

    static constexpr ck_tile::index_t kN0        = 128;
    static constexpr ck_tile::index_t kQKHeaddim = 96;
    static constexpr ck_tile::index_t kVHeaddim  = 96;

    using Type = ck_tile::TileFmhaBwdShape<FmhaBlockTile,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps2,
                                           WarpTile0,
                                           0 /* kMaxSeqLenQ */>;
};

// hd128 — bm0=16 (q-seqlen block) differs from the 64/96 presets.
template <>
struct HstuBwdShape<128>
{
    using FmhaBlockTile = ck_tile::sequence<16, 128, 128, 16, 128, 16, 32, 128, 128>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<1, 4, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

    static constexpr ck_tile::index_t kN0        = 128;
    static constexpr ck_tile::index_t kQKHeaddim = 128;
    static constexpr ck_tile::index_t kVHeaddim  = 128;

    using Type = ck_tile::TileFmhaBwdShape<FmhaBlockTile,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps2,
                                           WarpTile0,
                                           0 /* kMaxSeqLenQ */>;
};

// hd256 — bm0=16, bn0=64 (kN0). NOTE: bn0=64 means the determ split count
// (ceil(seqlen/kN0)) DOUBLES vs the 128-bn0 presets; the harness dq_acc
// workspace sizing must use kN0=64 for hd256 (see example_hstu_attention_bwd.cpp
// kN0_bwd) or the determ reduce overruns.
template <>
struct HstuBwdShape<256>
{
    using FmhaBlockTile = ck_tile::sequence<16, 64, 256, 16, 256, 16, 32, 256, 256>;
    using BlockWarps0   = ck_tile::sequence<1, 4, 1>;
    using BlockWarps1   = ck_tile::sequence<4, 1, 1>;
    using BlockWarps2   = ck_tile::sequence<1, 4, 1>;
    using WarpTile0     = ck_tile::sequence<16, 16, 32>;
    using WarpTile1     = ck_tile::sequence<16, 16, 16>;

    // NOTE kN0=64 (bn0) — half the other presets -> determ split count doubles; harness
    // kN0_bwd must read this (not a hardcoded 128) or the dq_acc workspace under-allocates.
    static constexpr ck_tile::index_t kN0        = 64;
    static constexpr ck_tile::index_t kQKHeaddim = 256;
    static constexpr ck_tile::index_t kVHeaddim  = 256;

    using Type = ck_tile::TileFmhaBwdShape<FmhaBlockTile,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps0,
                                           WarpTile0,
                                           BlockWarps1,
                                           WarpTile1,
                                           BlockWarps2,
                                           WarpTile0,
                                           0 /* kMaxSeqLenQ */>;
};
