// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"
#include "wmma_traits.hpp"

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

/*! @struct WmmaDefaultSelector
 * @brief Implements a default WMMA selector strategy for gfx11/12 target architectures.
 * This implements the K dimension search strategy to find the largest supported WMMA
 * instruction for the given M/N block sizes and datatypes.
 * @tparam DataTypeA Data type of matrix A
 * @tparam DataTypeB Data type of matrix B
 * @tparam DataTypeAcc Data type of the accumulator
 * @tparam BlockM Size of the M dimension
 * @tparam BlockN Size of the N dimension
 * @tparam BlockK Size of the K dimension
 * @tparam GfxTargetId Target architecture ID
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockKTest,
          uint32_t GfxTargetId>
struct WmmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, BlockM, BlockN, BlockK, GfxTargetId>
{
    private:
    // TODO: Move this power-of-2 check to a type_traits utility
    static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");
    static_assert(is_rdna_arch_id_v<GfxTargetId>,
                  "WmmaDefaultSelector only supports rdna target IDs");

    // By default, let's assume no special flags for WMMA
    using CtrlFlags = DefaultWmmaFlags<DataTypeA, DataTypeB, DataTypeAcc>;

    // Define our candidate WMMA implementation for the current parameters
    using CandidateOp = amdgcn_mma<DataTypeA,
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
                                     typename WmmaDefaultSelector<DataTypeA,
                                                                  DataTypeB,
                                                                  DataTypeAcc,
                                                                  BlockM,
                                                                  BlockN,
                                                                  BlockK / 2u,
                                                                  GfxTargetId>::SelectedOp>;
};

/*! @struct WmmaDefaultSelector
 * @brief Implements a default WMMA selector strategy for gfx11/12 target architectures.
 * This implements the K dimension == 1, which is the base case for the recursive K dimension
 * search. If no supported instruction is found, falls back to an unsupported pass-through
 * implementation.
 * @tparam DataTypeA Data type of matrix A
 * @tparam DataTypeB Data type of matrix B
 * @tparam DataTypeAcc Data type of the accumulator
 * @tparam BlockM Size of the M dimension
 * @tparam BlockN Size of the N dimension
 * @tparam GfxTargetId Target architecture ID
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t GfxTargetId>
struct WmmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, BlockM, BlockN, 1u, GfxTargetId>
{
    // By default, let's assume no special flags for WMMA
    using CtrlFlags = DefaultWmmaFlags<DataTypeA, DataTypeB, DataTypeAcc>;

    // Default unsupported pass-through if no instruction is found
    using SelectedOp =
        amdgcn_mma<DataTypeA, DataTypeB, DataTypeAcc, BlockM, BlockN, 1u, CtrlFlags, GfxTargetId>;
};

/*! @struct MmaDefaultSelector
 * @brief Implements the rdna default MMA selector strategy for wave-wise MMA decomposition.
 * This implements the M/N block size search strategy to find the largest supported WMMA
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
                          enable_if_rdna_target_id_t<GfxTargetId>>
{
    private:
    // Provide the default depth-K search strategy for each class of common WMMA shapes.
    // Start searching from the largest K dimension MFMA shape down to the smallest.
    using CandidateOp16x16 = typename WmmaDefaultSelector<DataTypeA,
                                                          DataTypeB,
                                                          DataTypeAcc,
                                                          16u,
                                                          16u,
                                                          128u,
                                                          GfxTargetId>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp =
        typename WmmaDefaultSelector<DataTypeA, DataTypeB, DataTypeAcc, 1u, 1u, 1u, GfxTargetId>::
            SelectedOp;

    // Traits for each candidate
    using CandidateTraits16x16 = MmaOpTraits<CandidateOp16x16>;

    // Check if each candidate is supported for the given fragment sizes
    // For this case, we require the fragment sizes to be multiples of the WMMA shape
    static constexpr IsSupported16x16 = CandidateTraits16x16::IsSupported && (FragM % 16u == 0u) &&
                                        (FragN % 16u == 0u) &&
                                        (CandidateTraits16x16::BlockK % 16u == 0u);

    public:
    // Select the largest supported WMMA operation for the given fragment shape
    using SelectedOp = conditional_t<IsSupported16x16, CandidateOp16x16, DefaultOp>;
};

} // namespace ck_tile::core::arch::mma
