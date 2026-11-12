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

} // namespace detail

} // namespace ck_tile
