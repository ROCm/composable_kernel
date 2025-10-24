// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"
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

/*! @struct MfmaDefaultSelector
 * @brief Implements a default MFMA selector strategy for gfx9 target architectures.
 * This implements the K dimension search strategy to find the largest supported MFMA
 * instruction for the given M/N block sizes and datatypes.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @param DataTypeA Data type of matrix A
 * @param DataTypeB Data type of matrix B
 * @param DataTypeAcc Data type of the accumulator
 * @param BlockM Block M dimension size
 * @param BlockN Block N dimension size
 * @param BlockKTest Current Block K dimension size to test
 * @param GfxTargetId Target architecture ID
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockKTest,
          uint32_t GfxTargetId>
struct MfmaDefaultSelector
{
    private:
    // TODO: Move this power-of-2 check to a type_traits utility
    static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");
    static_assert(is_gfx9_arch_id_v<GfxTargetId>,
                  "MfmaDefaultSelector only supports gfx9 target IDs");

    // By default, let's assume no special flags for MFMA
    using CtrlFlags = DefaultMfmaCtrlFlags;

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
    // Otherwise, test another smaller BlockK. If no existing implementations, we will get BlockK=0u
    // and fall back to the unsupported pass-through implementation.
    using SelectedOp = conditional_t<CandidateTraits::IsSupported,
                                     CandidateOp,
                                     typename MfmaDefaultSelector<DataTypeA,
                                                                  DataTypeB,
                                                                  DataTypeAcc,
                                                                  BlockM,
                                                                  BlockN,
                                                                  BlockK / 2u,
                                                                  GfxTargetId>::SelectedOp>;
};

/*! @struct MfmaDefaultSelector
 * @brief Implements the base case for the default MFMA selector when no supported instruction is
 * found.
 * @param DataTypeA Data type of matrix A
 * @param DataTypeB Data type of matrix B
 * @param DataTypeAcc Data type of the accumulator
 * @param BlockM Block M dimension size
 * @param BlockN Block N dimension size
 * @param BlockKTest Current Block K dimension size to test
 * @param GfxTargetId Target architecture ID
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockKTest,
          uint32_t GfxTargetId>
struct MfmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, BlockM, BlockN, 1u, GfxTargetId>
{
    // Default unsupported pass-through if no instruction is found
    using SelectedOp =
        amdgcn_mma<DataTypeA, DataTypeB, DataTypeA, BlockM, BlockN, 1u, CtrlFlags, GfxTargetId>;
};

/*! @struct MmaDefaultSelector
 * @brief Implements the gfx9 default MMA selector strategy for wave-wise MMA decomposition.
 * This implements the M/N block size search strategy to find the largest supported MFMA
 * instruction for the given datatypes.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @param DataTypeA Data type of matrix A
 * @param DataTypeB Data type of matrix B
 * @param DataTypeAcc Data type of the accumulator
 * @param FragM Size of the M dimension of the fragment to decompose
 * @param FragN Size of the N dimension of the fragment to decompose
 * @param FragK Size of the K dimension of the fragment to decompose
 * @param GfxTargetId Target architecture ID
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          uint32_t GfxTargetId>
struct MmaDefaultSelector<DataTypeA,
                          DataTypeB,
                          DataTypeAcc,
                          FragM,
                          FragN,
                          FragK,
                          GfxTargetId,
                          enable_if_gfx9_target_id_t<GfxTargetId>>
{
    private:
    // Provide the default depth-K search strategy for each class of common MFMA shapes.
    // Start searching from the largest K dimension MFMA shape down to the smallest.
    using CandidateOp4x4 =
        typename MfmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, 4u, 4u, 4u, GfxTargetId>::
            SelectedOp;
    using CandidateOp16x16 = typename MfmaDefaultSelector<DataTypeA,
                                                          DataTypeB,
                                                          DataTypeAcc,
                                                          16u,
                                                          16u,
                                                          128u,
                                                          GfxTargetId>::SelectedOp;
    using CandidateOp32x32 = typename MfmaDefaultSelector<DataTypeA,
                                                          DataTypeB,
                                                          DataTypeAcc,
                                                          32u,
                                                          32u,
                                                          64u,
                                                          GfxTargetId>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp =
        typename MfmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, 1u, 1u, 1u, GfxTargetId>::
            SelectedOp;

    // Traits for each candidate
    using CandidateTraits4x4   = MmaOpTraits<CandidateOp4x4>;
    using CandidateTraits16x16 = MmaOpTraits<CandidateOp16x16>;
    using CandidateTraits32x32 = MmaOpTraits<CandidateOp32x32>;

    // Check if each candidate is supported for the given fragment sizes
    // For this case, we require the fragment sizes to be multiples of the MFMA shape
    static constexpr IsSupported4x4 = CandidateTraits4x4::IsSupported && (FragM % 4u == 0u) &&
                                      (FragN % 4u == 0u) && (CandidateTraits4x4::BlockK % 4u == 0u);
    static constexpr IsSupported16x16 = CandidateTraits16x16::IsSupported && (FragM % 16u == 0u) &&
                                        (FragN % 16u == 0u) &&
                                        (CandidateTraits16x16::BlockK % 16u == 0u);
    static constexpr IsSupported32x32 = CandidateTraits32x32::IsSupported && (FragM % 32u == 0u) &&
                                        (FragN % 32u == 0u) &&
                                        (CandidateTraits32x32::BlockK % 32u == 0u);

    public:
    // Select the largest supported MFMA operation for the given fragment shape
    using SelectedOp =
        conditional_t<IsSupported32x32,
                      CandidateOp32x32,
                      conditional_t<IsSupported16x16,
                                    CandidateOp16x16,
                                    conditional_t<IsSupported4x4, CandidateOp4x4, DefaultOp>>>;
};

} // namespace ck_tile::core::arch::mma
