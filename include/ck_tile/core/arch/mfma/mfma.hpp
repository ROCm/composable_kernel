// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "../mma_common.hpp"

namespace ck_tile::core::arch::mfma
{

    /*! \struct amdgcn_mfma (to be used as a MfmaOp policy)
    *  \brief  Light builtin wrapper for mfma instructions. This class's job is to
    *          provide a uniform interface to invoke the appropriate mfma instruction
    *          based on the template parameters provided. This interface is to bridge
    *          the gap between the ck_tile API types and the native __builtin types.
    *  @tparam DataTypeTA Datatype of input A
    *  @tparam DataTypeTB Datatype of input B
    *  @tparam ComputeT Datatype of accumulator
    *  @tparam BlockM M-dimension of mfma block
    *  @tparam BlockN N-dimension of mfma block
    *  @tparam BlockK K-dimension of mfma block
    *  @tparam ABID MFMA ABID control flag
    *  @tparam BLGP MFMA BLGP control flag
    *  @tparam CBSZ MFMA CBSZ control flag
    *  @tparam GfxTarget The current gfx family target of interest being compiled
    *  @tparam TargetEnable Enabler for the current target if supported
    */
    // TODO: Do we really need to have the flags as template parameters here?
    template <typename DataTypeA,
            typename DataTypeB,
            typename ComputeT,
            uint32_t BlockM,
            uint32_t BlockN,
            uint32_t BlockK,
            uint32_t Cbsz = 0, // CBSZ flag, default 0
            uint32_t Abid = 0, // ABID flag, default 0
            uint32_t Blgp = 0, // BLGP flag, default 0
            uint32_t GfxTargetId = amdgcn_target_arch_id::CURRENT_ARCH_ID,
            typename Enabler     = void>
    struct amdgcn_mfma
    {
        // This is a pass-through implementation that isn't supported, and doesn't
        // do anything practical. The following trait will allow us to identify
        // unsupported instances, as we won't include it in the overloads to follow.
        using Unsupported = Unsupported;
    
        // Interface types for A, B, C vectors types
        using AVecType = ext_vector_t<DataTypeA, BlockM * BlockK / amdgcn_target_arch_id::WAVE_SIZE>;
        using BVecType = ext_vector_t<DataTypeB, BlockN * BlockK / amdgcn_target_arch_id::WAVE_SIZE>;
        using CVecType = ext_vector_t<ComputeT, BlockM * BlockN / amdgcn_target_arch_id::WAVE_SIZE>;

        // Execute the mfma operation
        CK_TILE_DEVICE static auto const& exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
        {
            return regsC; // No-op, just return C
        }
    };

} // namespace ck_tile::core::arch::mfma

// Include the architecture-specific MFMA implementations and traits
#include "mfma_gfx9.hpp"
#include "mfma_traits.hpp"
