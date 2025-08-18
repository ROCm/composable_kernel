// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "mfma_traits.hpp"

namespace ck_tile::core::arch::wmma
{
    // Intended for use behind the Mma interface, we will have access to common selection parameters
    // such as datatypes and block sizes. Based on these, we will attempt to select the most
    // appropriate wmma backend implementation that is supported on the current target architecture.
    // For example, we may want to search for the largest K dimension that is supported for given M/N
    // sizes, as this may yield better performance. We can use the wmma_traits to help with this
    // selection process.
    // NOTE: This is a recursive template structure that will attempt to find a supported wmma
    // implementation by adjusting the BlockK dimension downwards by powers of 2 until a match is 
    // found or we reach a base case (which does nothing).
    // We can write more sophisticated search strategies as needed with different selector classes.
    template<typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN,
             uint32_t BlockKTest = 64u> // Current max possible K-value for backend instr (most efficient)
    struct WmmaSelector
    {
        private:
        static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");

        // Define our candidate wmma implementation for the current parameters
        using CandidateOp = amdgcn_wmma<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>;
        using CandidateTraits = wmma_traits<CandidateOp>;

        // Dispatch our next search parameters, should the candidate not be supported.
        // NOTE: this is up to the library's needs, however we can give a simple example here and
        // search for a logical next smaller K dimension (assuming K is a power-of-2).
        constexpr static uint32_t NextBlockM = BlockM; // Keep M the same
        constexpr static uint32_t NextBlockN = BlockN; // Keep N the same
        constexpr static uint32_t NextBlockK = BlockK / 2u; // Search for smaller K

        public:
        // If the candidate is supported (e.g., a backend implementation exists), then select it.
        // Otherwise, test another smaller BlockK. If no existing implementations, keep the current
        // candidate.
        using SelectedOp = conditional_t<CandidateTraits::is_supported,
                                         CandidateOp,
                                         typename WmmaSelector<InputTA, InputTB, ComputeT, NextBlockM, NextBlockN, NextBlockK>::SelectedOp>;
    };

    template<typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN>
    struct WmmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>
    {
        // Mma_impl will just be a pass-through if no instruction is found
        using SelectedOp = amdgcn_wmma<InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>;
    };

} // namespace ck_tile::core::arch::wmma
