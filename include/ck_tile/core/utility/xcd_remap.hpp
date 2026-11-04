// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

// Number of XCDs the workgroups are dispatched round-robin across, for the kernels
// that remap their grid with remap_xcd_grid. 0 disables the remap. Only enabled
// where it has been measured (gfx1250); define it to override.
#if !defined(CK_TILE_XCD_REMAP_NUM_XCDS)
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
#define CK_TILE_XCD_REMAP_NUM_XCDS 8
#else
#define CK_TILE_XCD_REMAP_NUM_XCDS 0
#endif
#endif

namespace ck_tile {

// Workgroup id L is dispatched to XCD L % num_xcds, which scatters consecutive
// ids over every XCD. remap_xcd turns the ids each XCD receives into one
// contiguous segment of [0, total): segment x starts at
// x * short_len + min(x, tall_xcds) and holds short_len + 1 ids on the first
// tall_xcds XCDs, short_len on the others. For total = 16, num_xcds = 8:
//
//   id:          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
//   remap_xcd:   0  2  4  6  8 10 12 14  1  3  5  7  9 11 13 15
//
// so XCD 0 (ids 0, 8) works on 0 and 1, XCD 1 (ids 1, 9) on 2 and 3, and so on.
// It is a bijection on [0, total); remap_xcd_inverse is its inverse.
CK_TILE_HOST_DEVICE constexpr index_t remap_xcd(index_t id, index_t total, index_t num_xcds)
{
    const index_t short_len = total / num_xcds;
    const index_t tall_xcds = total - short_len * num_xcds;
    const index_t xcd       = id % num_xcds;
    return xcd * short_len + (xcd < tall_xcds ? xcd : tall_xcds) + id / num_xcds;
}

CK_TILE_HOST_DEVICE constexpr index_t
remap_xcd_inverse(index_t remapped, index_t total, index_t num_xcds)
{
    const index_t short_len = total / num_xcds;
    const index_t tall_xcds = total - short_len * num_xcds;
    const index_t tall_span = tall_xcds * (short_len + 1);
    if(remapped < tall_span)
        return (remapped % (short_len + 1)) * num_xcds + remapped / (short_len + 1);
    // Past the tall segments short_len >= 1 for every remapped < total.
    const index_t r = remapped - tall_span;
    return (r % short_len) * num_xcds + tall_xcds + r / short_len;
}

#if CK_TILE_XCD_REMAP_NUM_XCDS
// remap_xcd over the whole grid: linear_id is the workgroup's dispatch-order id,
// blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z).
CK_TILE_DEVICE index_t remap_xcd_grid(index_t linear_id)
{
    return remap_xcd(linear_id, gridDim.x * gridDim.y * gridDim.z, CK_TILE_XCD_REMAP_NUM_XCDS);
}
#endif

} // namespace ck_tile
