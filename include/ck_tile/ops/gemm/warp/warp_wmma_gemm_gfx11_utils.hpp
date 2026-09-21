// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// C distribution of gfx11 WMMA is not compatible with A distribution:
// C: 2 lanes per row (lane and lane + 16), 8 values per lane are interleaved.
// A: 1 lane per row, 16 values, lane and lane + 16 have the same values.
// This function transforms one ditribution to another for GEMM-GEMM scenarios.
template <typename OutTensor, typename InTensor>
CK_TILE_DEVICE static constexpr void PermuteWarpGemmCToA(OutTensor& out, const InTensor& in)
{
#if defined(__gfx11__)
    static_assert(sizeof(typename OutTensor::DataType) == 2);
    static_assert(std::is_same_v<typename OutTensor::DataType, typename InTensor::DataType>);

    constexpr index_t n_out = OutTensor::get_thread_buffer_size();
    static_assert(n_out == InTensor::get_thread_buffer_size() * 2);

    // Perm byte selectors are swapped for the second row (16 lanes) because it needs to be done
    // once instead to swapping w and v everytime
    const uint32_t byte_selector0 = get_lane_id() < 16 ? 0x05'04'01'00 : 0x01'00'05'04;
    const uint32_t byte_selector1 = get_lane_id() < 16 ? 0x07'06'03'02 : 0x03'02'07'06;
    static_for<0, n_out, 1>{}([&](auto i) {
        const auto v = in.get_thread_buffer().template get_as<uint32_t>(i);
        // Swap rows (lane <-> lane ^ 16)
        const auto w = __builtin_amdgcn_permlanex16(0, v, 0x76543210, 0xfedcba98, false, true);
        // Interleave values of lane and lane ^ 16
        out.get_thread_buffer().template set_as<uint32_t>(
            number<i * 2 + 0>{}, __builtin_amdgcn_perm(w, v, byte_selector0));
        out.get_thread_buffer().template set_as<uint32_t>(
            number<i * 2 + 1>{}, __builtin_amdgcn_perm(w, v, byte_selector1));
    });
#else
    static_assert(false, "PermuteWarpGemmCToA is only for gfx11");
    ignore = out;
    ignore = in;
#endif
}

} // namespace ck_tile
