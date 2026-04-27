// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class SparseWmmaDefaultSelector
 * @brief Implements a default sparse WMMA selector strategy. The SelectedOp can be unsupported.
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      Size of the M dimension
 * @tparam WaveTileN      Size of the N dimension
 * @tparam WaveTileKTest  Size of the K dimension
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileKTest,
          typename CompilerTarget>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_target_arch_rdna(CompilerTarget) &&
// is_power_of_two_integer(WaveTileKTest))
struct SparseWmmaDefaultSelector
{
    private:
    // Define our candidate WMMA implementation for the current parameters
    using CandidateOp = amdgcn_mma<ADataType,
                                   BDataType,
                                   CDataType,
                                   WaveTileM,
                                   WaveTileN,
                                   WaveTileKTest,
                                   DefaultWmmaCtrlFlags,
                                   CompilerTarget,
                                   MmaOpFamily::SPARSE>;

    using CandidateTraits = MmaOpTraits<CandidateOp>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<CandidateTraits::IsSupported,
                                          CandidateOp,
                                          amdgcn_mma<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     WaveTileM,
                                                     WaveTileN,
                                                     WaveTileKTest,
                                                     void,
                                                     amdgcn_target<>,
                                                     MmaOpFamily::UNDEFINED>>;
};

/**
 * @struct MmaDefaultSelector
 * @brief Implements the RDNA default MMA selector strategy for sparse WMMA.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      Size of the M dimension of the WaveTile to decompose
 * @tparam WaveTileN      Size of the N dimension of the WaveTile to decompose
 * @tparam WaveTileK      Size of the K dimension of the WaveTile to decompose
 * @tparam CompilerTarget The compiler target
 * @tparam OpFamily       The MMA operation family
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          typename CompilerTarget,
          MmaOpFamily OpFamily>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          WaveTileM,
                          WaveTileN,
                          WaveTileK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<enable_if_target_family_gfx12_t<CompilerTarget>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::SPARSE>>>
{
    private:
    // Provide the default depth-K search strategy for each class of common WMMA shapes.
    // Start searching from the largest K dimension WMMA shape down to the smallest.
    using CandidateOp16x16 = typename SparseWmmaDefaultSelector<ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                16u,
                                                                16u,
                                                                32u,
                                                                CompilerTarget>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp = typename SparseWmmaDefaultSelector<ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         1u,
                                                         1u,
                                                         1u,
                                                         CompilerTarget>::SelectedOp;

    // Check if each candidate is supported for the given WaveTile sizes
    // For this case, we require the WaveTile sizes to be multiples of the WMMA shape
    static constexpr bool IsSupported16x16 =
        MmaOpTraits<CandidateOp16x16>::IsSupported && (WaveTileM % CandidateOp16x16::kM == 0u) &&
        (WaveTileN % CandidateOp16x16::kN == 0u) && (WaveTileK % CandidateOp16x16::kK == 0u);

    public:
    // Select the largest supported WMMA operation for the given WaveTile shape
    using SelectedOp = std::conditional_t<IsSupported16x16, CandidateOp16x16, DefaultOp>;
};

} // namespace ck_tile::core::arch::mma
