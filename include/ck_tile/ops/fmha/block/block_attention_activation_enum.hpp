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

// The constant factor attention_activation_x2 can leave to the caller: act(scale x) =
// attention_activation_out_factor(scale) * act'(x).
template <BlockAttentionActivationEnum Activation>
CK_TILE_DEVICE constexpr float attention_activation_out_factor(float scale)
{
    static_assert(Activation == BlockAttentionActivationEnum::FAST_GELU,
                  "attention_activation_out_factor: an activation without an implementation");
    return 0.5f * scale;
}

// p[i] = attention_activation(s[i], scale) over two equally long thread buffers of fp32, two
// scores at a time on packed fp32 and in three passes (on gfx125: no hardware tanh waits on its
// own input or feeds the next instruction). kOutFactor leaves out
// attention_activation_out_factor(scale) for the caller to take once on the gemm's output: p is
// then 2 / scale times larger, so a 16-bit P needs the range of bf16 for it.
template <BlockAttentionActivationEnum Activation, bool kOutFactor, typename SBuf, typename PBuf>
CK_TILE_DEVICE void attention_activation_x2(const SBuf& s, PBuf& p, float scale)
{
    static_assert(Activation == BlockAttentionActivationEnum::FAST_GELU,
                  "attention_activation_x2: an activation without an implementation");
    constexpr index_t kN = SBuf::size();
    static_assert(kN % 2 == 0 && PBuf::size() == kN);
#if defined(__gfx125__)
    // y = (h x) (1 + tanh(u)), u = x (a0 + a1 x^2) as in attention_activation; h = 1 with
    // kOutFactor, else its 0.5 scale
    constexpr float c0 = 0.7978845608028654f;
    constexpr float c1 = c0 * 0.044715f;
    const float a1s    = c1 * scale * scale * scale;
    const fp32x2_t a0  = {c0 * scale, c0 * scale};
    const fp32x2_t a1  = {a1s, a1s};
    const float hs     = attention_activation_out_factor<Activation>(scale);
    const fp32x2_t h   = {hs, hs};
    static_for<0, kN / 2, 1>{}([&](auto i) {
        const fp32x2_t x       = {s[number<2 * i>{}], s[number<2 * i + 1>{}]};
        const fp32x2_t u       = x * (a1 * (x * x) + a0);
        p(number<2 * i>{})     = u[0];
        p(number<2 * i + 1>{}) = u[1];
    });
    static_for<0, kN, 1>{}([&](auto i) { p(i) = __builtin_amdgcn_tanhf(p[i]); });
    static_for<0, kN / 2, 1>{}([&](auto i) {
        fp32x2_t x = {s[number<2 * i>{}], s[number<2 * i + 1>{}]};
        if constexpr(!kOutFactor)
            x = x * h;
        const fp32x2_t t       = {p[number<2 * i>{}], p[number<2 * i + 1>{}]};
        const fp32x2_t y       = x * t + x;
        p(number<2 * i>{})     = y[0];
        p(number<2 * i + 1>{}) = y[1];
    });
#else
    const float r = kOutFactor ? 1.f / attention_activation_out_factor<Activation>(scale) : 1.f;
    static_for<0, kN, 1>{}(
        [&](auto i) { p(i) = attention_activation<Activation>(s[i], scale) * r; });
#endif
}

} // namespace ck_tile
