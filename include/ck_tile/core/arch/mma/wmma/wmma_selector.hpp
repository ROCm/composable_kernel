// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "mfma_traits.hpp"

namespace ck_tile::core::arch::mma {
/*! @struct DefaultWmmaFlags
 * @brief Generates default WMMA control flags based on data types.
 * @tparam DataTypeA Data type of matrix A
 * @tparam DataTypeB Data type of matrix B
 * @tparam DataTypeAccum Data type of the accumulator
 */
template <typename DataTypeA, typename DataTypeB, typename DataTypeAccum>
struct DefaultWmmaFlags
{
    // Generate default flags for signage
    // Only used currently for integer inputs / accum in gfx11 / gfx12
    constexpr static WmmaCtrlFlags InputSignA =
        std::is_signed_v<DataTypeA> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags InputSignB =
        std::is_signed_v<DataTypeB> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags AccumSign =
        std::is_signed_v<DataTypeAccum> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;

    // Generate default flags for accumulator destination bits.
    // Only used if accumulation size is 16-bit in gfx11
    constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
};

/*! @struct MmaDefaultSelector
 * @brief Implements a default WMMA selector strategy for gfx11/12 target architectures.
 * This implements the K dimension search strategy to find the largest supported WMMA
 * instruction for the given M/N block sizes and datatypes.
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockKTest,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<DataTypeA,
                          DataTypeB,
                          DataTypeAcc,
                          BlockM,
                          BlockN,
                          BlockK,
                          GfxTargetId,
                          enable_if_rdna_target_id_t<GfxTargetId>>
{
    private:
    static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");

    // By default, let's assume no special flags for MFMA
    using CtrlFlags = DefaultWmmaFlags<DataTypeA, DataTypeB, DataTypeAcc>;

    // Define our candidate MFMA implementation for the current parameters
    using CandidateOp     = amdgcn_mma<DataTypeA,
                                       DataTypeB,
                                       DataTypeAcc,
                                       BlockM,
                                       BlockN,
                                       BlockKTest,
                                       CtrlFlags,
                                       GfxTargetId>;
    using CandidateTraits = MmaOpTraits<CandidateOp>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, test another smaller BlockK. If no existing implementations, keep the current
    // candidate.
    using SelectedOp = conditional_t<CandidateTraits::IsSupported,
                                     CandidateOp,
                                     typename MmaDefaultSelector<DataTypeA,
                                                                 DataTypeB,
                                                                 DataTypeAcc,
                                                                 BlockM,
                                                                 BlockN,
                                                                 BlockK / 2u>::SelectedOp>;
};

template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<DataTypeA,
                          DataTypeB,
                          DataTypeAcc,
                          BlockM,
                          BlockN,
                          0u,
                          GfxTargetId,
                          enable_if_gfx9_target_id_t<GfxTargetId>>
{
    // Default unsupported pass-through if no instruction is found
    using SelectedOp =
        amdgcn_mma<DataTypeA, DataTypeB, DataTypeAcc, BlockM, BlockN, 0u, GfxTargetId>;
};

} // namespace ck_tile::core::arch::mma
