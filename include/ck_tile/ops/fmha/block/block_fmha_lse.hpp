// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// The LSE of a softmax row from its running max m_log2 (log2 units: scale_s and log2 e applied)
// and its sum l of 2^(s - m_log2): the natural log, or with lse_log2 (fmha_fwd_args::lse_log2)
// base 2, log2 of the row sum of 2^(s log2 e), computed as such.
CK_TILE_DEVICE float fmha_lse(float m_log2, float l, bool lse_log2)
{
    // natural: (m_log2 + log2(l)) * ln(2), a v_log_f32 and an fma
    const float lse = m_log2 + log2f(l);
    return lse_log2 ? lse : lse * 0.693147180559945309f;
}

// An LSE known in natural log (a sink's), in the base fmha_lse gives.
CK_TILE_DEVICE float fmha_lse_from_natural(float lse, bool lse_log2)
{
    return lse_log2 ? lse * static_cast<float>(C_LOG2E) : lse;
}

} // namespace ck_tile
