// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// What turns the scores S = scale_s * Q K^T into the P of O = P V.
//
// SOFTMAX is attention as usual: P = softmax(S), row by row.
// The others are generalized dot-product attention (GDPA): P = act(S) elementwise, with no
// row max, no row sum and no normalization of O, so no LSE either. A masked or padded key
// contributes nothing (P = 0) whatever act(-inf) would be.
//   FAST_GELU: gelu with the tanh approximation, x * sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)),
//              which is 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) exactly
enum class BlockAttentionActivationEnum
{
    SOFTMAX   = 0,
    FAST_GELU = 1,
};

namespace detail {
// T::Activation where T has one (TileFmhaTraits, BlockFmhaPipelineProblem), softmax otherwise
template <typename T, typename = void>
inline constexpr auto fmha_activation_v = BlockAttentionActivationEnum::SOFTMAX;
template <typename T>
inline constexpr auto fmha_activation_v<T, std::void_t<decltype(T::Activation)>> = T::Activation;
} // namespace detail

// The pipelines mask a key out of GDPA with a score of 0 instead of zeroing P, which needs
// act(0) = 0: an activation states it here.
template <BlockAttentionActivationEnum>
inline constexpr bool attention_activation_zero_at_zero_v = false;
template <>
inline constexpr bool attention_activation_zero_at_zero_v<BlockAttentionActivationEnum::FAST_GELU> =
    true;

// What a pipeline needs to know of its activation, in one place:
// - is_softmax: softmax, or GDPA (P = act(scale_s * S), no row max/sum, no normalization of O);
// - masked_score: the score a masked or padded key gets, -inf for softmax and, for GDPA, 0,
//   which the activation maps to 0 (act(0) = 0, checked here), so P needs no check for it.
// GDPA's scores take no exp2, so a GDPA pipeline also tells the kernel to pass scale_s as is
// (kIgnoreFastExp2 = !is_softmax).
template <BlockAttentionActivationEnum Activation>
struct BlockAttentionActivationTraits
{
    static constexpr bool is_softmax = Activation == BlockAttentionActivationEnum::SOFTMAX;
    static_assert(is_softmax || attention_activation_zero_at_zero_v<Activation>,
                  "GDPA masks keys with a score of 0, so act(0) must be 0");
    static constexpr float masked_score = is_softmax ? -numeric<float>::infinity() : 0.f;
};

// act(scale * x) of a GDPA activation, in fp32. scale is uniform: its products below are
// hoisted out of the caller's loops. FAST_GELU is ck_tile::element_wise::FastGelu's tanh GELU,
// written for this hot loop: scale folded into its constants and, on gfx125, the hardware tanh
// (one transcendental), where FastGelu takes an exp and a division.
template <BlockAttentionActivationEnum Activation>
CK_TILE_DEVICE float attention_activation(float x, float scale)
{
    static_assert(Activation != BlockAttentionActivationEnum::SOFTMAX,
                  "softmax is not elementwise");
    if constexpr(Activation == BlockAttentionActivationEnum::FAST_GELU)
    {
        // y = scale x; gelu_tanh(y) = h y (1 + tanh(u)), h = 1/2, u = sqrt(2/pi) (y + 0.044715 y^3)
        constexpr float c0 = 0.7978845608028654f;
        constexpr float c1 = c0 * 0.044715f;
        const float a0     = c0 * scale;                 // u = x (a0 + a1 x^2)
        const float a1     = c1 * scale * scale * scale; //
        const float hs     = 0.5f * scale;
        const float u      = x * __builtin_fmaf(a1, x * x, a0);
        const float hy     = hs * x;
#if defined(__gfx125__)
        // one transcendental: the hardware tanh
        return __builtin_fmaf(hy, __builtin_amdgcn_tanhf(u), hy);
#else
        // tanh(u) = 1 - 2 / (1 + e^(2u)), as 2 hy sigmoid(2u)
        return 2.f * hy * __builtin_amdgcn_rcpf(1.f + exp2(-2.f * 1.4426950408889634f * u));
#endif
    }
    else
    {
        static_assert(Activation == BlockAttentionActivationEnum::FAST_GELU,
                      "attention_activation: an activation without an implementation");
    }
}

} // namespace ck_tile
