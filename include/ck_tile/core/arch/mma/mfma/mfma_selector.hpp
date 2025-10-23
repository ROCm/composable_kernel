// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "mfma_traits.hpp"

namespace ck_tile::core::arch::mma {
/*! @struct DefaultMmaCtrlFlags
 * @brief Default MFMA flags, no broadcasting or rotation of inputs
 */
struct DefaultMfmaCtrlFlags
{
    static constexpr uint32_t Cbsz = 0; // CBSZ flag, default 0
    static constexpr uint32_t Abid = 0; // ABID flag, default 0
    static constexpr uint32_t Blgp = 0; // BLGP flag, default 0
};

/*! @struct MmaDefaultSelector
 * @brief Implements a default MFMA selector strategy for gfx9 target architectures.
 * This implements the K dimension search strategy to find the largest supported MFMA
 * instruction for the given M/N block sizes and datatypes.
 */
template <typename InputTA,
          typename InputTB,
          typename ComputeT,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockKTest,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<InputTA,
                          InputTB,
                          ComputeT,
                          BlockM,
                          BlockN,
                          BlockK,
                          GfxTargetId,
                          enable_if_gfx9_target_id_t<GfxTargetId>>
{
    private:
    static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");

    // By default, let's assume no special flags for MFMA
    using CtrlFlags = DefaultMfmaCtrlFlags;

    // Define our candidate MFMA implementation for the current parameters
    using CandidateOp =
        amdgcn_mma<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest, CtrlFlags, GfxTargetId>;
    using CandidateTraits = MmaOpTraits<CandidateOp>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, test another smaller BlockK. If no existing implementations, keep the current
    // candidate.
    using SelectedOp = conditional_t<
        CandidateTraits::IsSupported,
        CandidateOp,
        typename MmaDefaultSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK / 2u>::
            SelectedOp>;
};

template <typename InputTA,
          typename InputTB,
          typename ComputeT,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<InputTA,
                          InputTB,
                          ComputeT,
                          BlockM,
                          BlockN,
                          0u,
                          GfxTargetId,
                          enable_if_gfx9_target_id_t<GfxTargetId>>
{
    // Default unsupported pass-through if no instruction is found
    using SelectedOp = amdgcn_mma<InputTA, InputTB, ComputeT, BlockM, BlockN, 0u, GfxTargetId>;
};

} // namespace ck_tile::core::arch::mma
