// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <stdexcept>
#include <string>
#include <cassert>

namespace ck_tile {

template <typename BlockShape_>
struct FusedMoeGemmTilePartitioner_NoAtomic
{
    using BlockShape = ck_tile::remove_cvref_t<BlockShape_>;

    static constexpr const char* name = "no_atomic";

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*num_sorted_tiles*/,
                                   ck_tile::index_t intermediate_size)
    {
        // Validate that workgroup can handle full intermediate dimension
        // This eliminates the need for K-splitting and atomics
        
        // Use intermediate_size in a way that compiler always recognizes
        (void)intermediate_size; // Suppress unused parameter warning
        
        // Runtime validation - this will always use the parameter
        if (intermediate_size > BlockShape::Block_N0) {
            // This forces the compiler to see intermediate_size as used
            // In device code, we can't throw, so we use a device assert
#ifdef __CUDA_ARCH__ 
            __trap(); // CUDA device trap
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_trap(); // HIP device trap  
#else
            assert(false && "intermediate_size too large for no-atomic approach");
#endif
        }
        
        // Key change: partition experts along token dimension only
        // Each workgroup handles a full intermediate_size slice
        index_t i_m = blockIdx.y;  // Expert/token tile (since grid is dim3(1, ms, 1))
        index_t i_n = blockIdx.x;  // Will always be 0 since ns=1 (no K-splitting)
        
        return ck_tile::make_tuple(i_m, i_n);
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t max_tokens, index_t intermediate_size)
    {
        // Validate that workgroup can handle the full intermediate_size
        // If this fails, consider using atomic approach or increasing Block_N0
        if (intermediate_size > BlockShape::Block_N0) {
            throw std::runtime_error("intermediate_size (" + std::to_string(intermediate_size) + 
                                   ") > Block_N0 (" + std::to_string(BlockShape::Block_N0) + 
                                   "). Cannot use no-atomic approach.");
        }
        
        // Calculate grid dimensions
        index_t ms = ck_tile::integer_divide_ceil(max_tokens, BlockShape::Block_M0);
        
        // KEY FIX: Since each workgroup handles full intermediate_size,
        // we only need 1 workgroup in the K dimension (no splitting)
        index_t ns = 1;  // No K-splitting - single workgroup handles entire intermediate_size
        
        // Grid layout: dim3(K-splits=1, M-splits, 1)
        // - blockIdx.x will always be 0 (single K-split)
        // - blockIdx.y ranges from 0 to ms-1 (token tiles)
        return dim3(ns, ms, 1);
    }
};

} // namespace ck_tile

