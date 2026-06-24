// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class MmaDefaultSelector
 * @brief Implements a default mma selector strategy for the current target architecture. This is
 * simply intended as a default selection strategy for mma instruction operations. Given the
 * particular datatypes and WaveTile dimensions, the selector will attempt to select the instruction
 * with the largest K dimension that is supported on the current target architecture.
 * @tparam ADataType       Data type of matrix A
 * @tparam BDataType       Data type of matrix B
 * @tparam CDataType       Data type of the accumulator
 * @tparam WaveTileM       WaveTile M dimension
 * @tparam WaveTileN       WaveTile N dimension
 * @tparam WaveTileK       WaveTile K dimension
 * @tparam CompilerTarget  The compiler target
 * @tparam OpFamily        The MMA operation family
 * @tparam Enable          SFINAE enabler
 * @note Here we distinguish that WaveTile MNK sizes from Fragment MNK sizes used in the actual MMA
 * operation. WaveTile sizes correspond to the overall tile size being computed, while Fragment
 * sizes correspond to the size of the individual MMA instructions being used to compute the overall
 * in fragment-wise. The WaveTile sizes must be multiples of the Fragment sizes and in general
 * larger than or equal to the Fragment sizes.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          typename CompilerTarget,
          MmaOpFamily OpFamily,
          typename Enable = void>
// TODO c++20 requires
struct MmaDefaultSelector
{
    // By default, no selection is made, and we fall back to a pass-through unsupported
    // implementation. This is because we do not have any knowledge of the target architecture here.
    using SelectedOp = amdgcn_mma<ADataType,
                                  BDataType,
                                  CDataType,
                                  WaveTileM,
                                  WaveTileN,
                                  WaveTileK,
                                  void,
                                  amdgcn_target<>,
                                  MmaOpFamily::UNDEFINED>;
};

#if CK_TILE_CONCEPTS

/**
 *  @concept MmaSelectorI
 *  @brief  Expresses the required members for each MmaSelector class.
 */
template <typename MmaSelector>
concept MmaSelectorI = requires(MmaSelector op) {
    // Selectors should have a resulting SelectedOp type
    typename MmaSelector::SelectedOp;
};

#endif // CK_TILE_CONCEPTS

/**
 * @class MmaKSearchSelector
 * @brief Generic helper for selecting the intrinsic with the largest possible K size that divides
 * WaveTileKTest. If nothing is supported, a generic unsupported passthrough impl is returned.
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      WaveTile M dimension size
 * @tparam WaveTileN      WaveTile N dimension size
 * @tparam WaveTileKTest  Current WaveTile K dimension size to test
 * @tparam CompilerTarget The compiler target
 * @note Here we assume that WaveTileKTest is always a power-of-two integer. The search strategy
 *       starts from a maximum WaveTileKTest size down to 1u by halving each time.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileKTest,
          typename CtrlFlags,
          typename CompilerTarget, // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
          MmaOpFamily OpFamily>
struct MmaKSearchSelector
{
    private:
    static_assert(!(WaveTileKTest & (WaveTileKTest - 1)), "WaveTileKTest must be power of 2.");

    using CandidateOp = amdgcn_mma<ADataType,
                                   BDataType,
                                   CDataType,
                                   WaveTileM,
                                   WaveTileN,
                                   WaveTileKTest,
                                   CtrlFlags,
                                   CompilerTarget,
                                   OpFamily>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, test another smaller WaveTileK. If no existing implementations, we will get
    // WaveTileK=0u and fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<MmaOpTraits<CandidateOp>::IsSupported,
                                          CandidateOp,
                                          typename MmaKSearchSelector<ADataType,
                                                                      BDataType,
                                                                      CDataType,
                                                                      WaveTileM,
                                                                      WaveTileN,
                                                                      WaveTileKTest / 2u,
                                                                      CtrlFlags,
                                                                      CompilerTarget,
                                                                      OpFamily>::SelectedOp>;
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          typename CtrlFlags,
          typename CompilerTarget, // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
          MmaOpFamily OpFamily>
struct MmaKSearchSelector<ADataType,
                          BDataType,
                          CDataType,
                          WaveTileM,
                          WaveTileN,
                          0u,
                          CtrlFlags,
                          CompilerTarget,
                          OpFamily>
{
    // Recursion endpoint: unsupported default implementation.
    using SelectedOp = amdgcn_mma<ADataType,
                                  BDataType,
                                  CDataType,
                                  1u,
                                  1u,
                                  1u,
                                  CtrlFlags,
                                  CompilerTarget,
                                  OpFamily>;
};

} // namespace ck_tile::core::arch::mma

// Include the implementations
#include "wmma/wmma_selector.hpp"
#include "mfma/mfma_selector.hpp"
#include "sparse/sparse_selector.hpp"
