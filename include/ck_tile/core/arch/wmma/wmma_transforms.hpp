// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

#include "mma_traits.hpp"

namespace ck::tile::core::arch::wmma
{
    // GFX11 specific transform for WMMA op: duplicate input data in upper / lower 16 lanes
    struct DuplicateTransformGfx11
    {
        template<typename VecType>
        CK_TILE_DEVICE static auto exec(VecType const& v)
        {
            // TODO: Implement swizzle duplication logic
            return v;
        }
    };

    // GFX11 specific transform for WMMA ops: pad C/D data to 32 bit wide accumulator
    struct PadTransformGfx11
    {
        template<typename VecType>
        CK_TILE_DEVICE static auto exec(VecType const& v)
        {
            // TODO: Implement b32 logic to pad 16->32 for gfx11
            return v;
        }
    };

    // GFX11 specific transform for WMMA ops: unpad C/D data from 32 bit wide accumulator
    struct UnpadTransformGfx11
    {
        template<typename VecType>
        CK_TILE_DEVICE static auto exec(VecType const& v)
        {
            // TODO: Implement b32 logic to unpad 32->16 for gfx11
            return v;
        }
    };

} // namespace ck::tile::core::arch::wmma
