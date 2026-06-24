// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

// TODO: We do not allow M / N composition for now, since it is not used in current CK Tile and the
// existing implementation was not able to deal with unusual MxN sizes such as for multi-block
// intrinsics.
/**
 * @struct MmaDefaultSelector
 * @brief Implements the CDNA default MMA selector strategy for sparse MFMA.
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
          MmaOpFamily OpFamily> // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          WaveTileM,
                          WaveTileN,
                          WaveTileK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<std::enable_if_t<is_any_value_of(CompilerTarget::TARGET_ID,
                                                                         amdgcn_target_id::GFX942,
                                                                         amdgcn_target_id::GFX950)>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::SPARSE>>>
{
    // Search for largest intrinsic K size that fits in WaveTileK.
    using SelectedOp = typename MmaKSearchSelector<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   WaveTileM,
                                                   WaveTileN,
                                                   WaveTileK,
                                                   DefaultSparseMfmaCtrlFlags,
                                                   CompilerTarget,
                                                   MmaOpFamily::SPARSE>::SelectedOp;
};
} // namespace ck_tile::core::arch::mma
