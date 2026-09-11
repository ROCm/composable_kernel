// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared types for all FMHA BWD kernel families (OGradDotO, DqDkDv, ConvertDq).
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files).

#pragma once

#include <rocm_ck/datatype.hpp>
#include <rocm_ck/index_t.hpp>

#include <cstdint>

namespace rocm_ck {

// Padding semantics vary per kernel family:
//   OGradDotO / ConvertDQ: bool (pad or no-pad)
//   DqDkDv: int {0=none, 1=small, 8=full vector-aligned}
// The tri-valued int maps to CK Tile's TileFmhaBwdTraits::kPadHeadDimQ/V
// which controls vector load widths. 0 = no padding, 1 = scalar fallback,
// 8 = full 128-bit vector loads with padding. bool is sufficient for
// OGradDotO/ConvertDQ which only need on/off.

/// FMHA attention mode: fixed-length batches vs variable-length groups.
enum class FmhaMode
{
    BATCH,
    GROUP
};

/// Bias type for attention score modification.
/// Values must match ck_tile::BlockAttentionBiasEnum.
enum class FmhaBiasType
{
    NONE,
    ELEMENTWISE,
    ALIBI
};

/// Attention mask family.
///
/// Integer values must match ck_tile::GenericAttentionMaskEnum so the device
/// bridge can forward the spec-time enum to the kernel via a static_cast
/// without a translation table (see dqdkdv_dev.hpp's MASK_TYPE scalar wiring).
///
/// Both causal variants describe a sliding window with left=-1 (unbounded
/// past) and right=0 (no lookahead). The two flavours differ in where the
/// causal diagonal is anchored when seqlen_q != seqlen_k:
///   * TOP_LEFT     -- diagonal at (0, 0); standard "predict next token".
///   * BOTTOM_RIGHT -- diagonal at (seqlen_q-1, seqlen_k-1); used when the
///                     query is the *tail* of a longer cached K/V (decode).
/// GENERIC selects ck_tile's runtime (left, right, top-left/bottom-right)
/// window descriptor and is intended for sliding-window / xformer-style
/// attention. NO_MASK disables masking at compile time.
enum class FmhaMaskType
{
    NO_MASK             = 0,
    TOP_LEFT_CAUSAL     = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    GENERIC             = 3,
};

} // namespace rocm_ck
