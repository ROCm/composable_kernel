// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "arch.hpp"
#include "mfma/mfma_traits.hpp"
#include "wmma/wmma_traits.hpp"

namespace ck::tile::core::arch
{
    namespace detail
    {
        template <typename MmaOp, typename Enabler = void>
        struct mma_traits;

        // Inherit selectively from mfma_traits or wmma_traits based on the MmaOp type.
        // Additional arch sanity check will give us breadcrumbs in the compiler issues if they arise.
        template <typename MfmaOp>
        struct mma_traits<MfmaOp, std::enable_if_t<is_mfma_op_v<MfmaOp>>>
        : public mfma_traits<MfmaOp> 
        {
            static_assert(CK_TILE_ARCH_GFX9, "MFMA is only supported on GFX9 architecture");
            static_assert(CK_TILE_WAVE64_MODE, "MFMA is only supports wave64 mode");
        };

        template <typename WmmaOp>
        struct mma_traits<WmmaOp, std::enable_if_t<is_wmma_op_v<WmmaOp>>>
        : public wmma_traits<WmmaOp> 
        {
            static_assert(CK_TILE_ARCH_GFX11 || CK_TILE_ARCH_GFX12, "WMMA is only supported on GFX11 / GFX12 architectures");
            static_assert(CK_TILE_WAVE32_MODE, "WMMA is only supports wave32 mode");
        };

    } // namespace detail

    // The following is the primary interface to query MMA operation traits in ck_tile API
    // TODO: use C++20 concepts here to ensure that the basic interface is present from MmaOp object
    // e.g., kM, kN, kK, ...:
    // kAMBLock, ...
    // AVecType, ...
    template <typename MmaOp>
    struct mma_traits : public detail::mma_traits<MmaOp> 
    {
        // Additional traits
        constexpr static bool IsMfma = is_mfma_op_v<MmaOp>;
        constexpr static bool IsWmma = is_wmma_op_v<MmaOp>;
    };

} // namespace ck::tile::core::arch
