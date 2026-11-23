// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"

namespace ck_tile {

namespace detail {

// Opt-in markers an fmha fwd pipeline may define to pick its K/V DRAM views:
// - kUsesUntransposedVKernelPath: V is consumed as (seqlen_k, hdim_v) instead
//   of being transposed to (hdim_v, seqlen_k).
// - kUsesTdmAffineDramPath: K/V views stay unpadded; the TDM box clamp
//   zero-fills reads past their bounds.
// A pipeline that declares neither gets Fallback<T>::value.
template <typename T>
using has_untransposed_v_kernel_path = decltype(T::kUsesUntransposedVKernelPath);

template <typename T>
using has_tdm_affine_dram_path = decltype(T::kUsesTdmAffineDramPath);

template <typename T>
struct kernel_path_off_by_default : std::false_type
{
};

template <typename T, template <typename> class Fallback = kernel_path_off_by_default>
inline constexpr bool uses_untransposed_v_kernel_path_v = []() {
    if constexpr(is_detected<has_untransposed_v_kernel_path, T>::value)
        return static_cast<bool>(T::kUsesUntransposedVKernelPath);
    else
        return Fallback<T>::value;
}();

template <typename T, template <typename> class Fallback = kernel_path_off_by_default>
inline constexpr bool uses_tdm_affine_dram_path_v = []() {
    if constexpr(is_detected<has_tdm_affine_dram_path, T>::value)
        return static_cast<bool>(T::kUsesTdmAffineDramPath);
    else
        return Fallback<T>::value;
}();

template <typename Pipeline, typename = void>
struct uses_qr_tdm_lds_arena : std::false_type
{
};

template <typename Pipeline>
struct uses_qr_tdm_lds_arena<Pipeline, std::void_t<decltype(Pipeline::kUsesLdsArena)>>
    : std::bool_constant<Pipeline::kUsesLdsArena>
{
};

template <typename Pipeline>
inline constexpr bool uses_qr_tdm_lds_arena_v = uses_qr_tdm_lds_arena<Pipeline>::value;

// Whether the K/V views take the NT_HT cache hint on gfx1250 (see
// FmhaFwdKernel::kKVCoherence): TDM pipelines that stage K/V in their LDS arena,
// fp16/bf16, head dims below 256, tiles of at most 64 rows.
template <typename Pipeline>
inline constexpr bool uses_kv_nt_ht_hint_v =
    uses_qr_tdm_lds_arena_v<Pipeline> && Pipeline::kM0 <= 64 &&
    Pipeline::BlockFmhaShape::kQKHeaddim < 256 &&
    (std::is_same_v<typename Pipeline::KDataType, half_t> ||
     std::is_same_v<typename Pipeline::KDataType, bf16_t>);

} // namespace detail

} // namespace ck_tile
