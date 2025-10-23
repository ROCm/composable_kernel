// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "arch.hpp"

namespace ck::tile::core::arch::mma {
/*! @struct MmaDefaultSelector
 * @brief Implements a default mma selector strategy for the current target architecture.
 * This is simply intended as a default selection strategy for mma instruction operations.
 * Given the particular datatypes and block M and N sizes, the selector will attempt to
 * select the instruction with the largest K dimension that is supported on the current target
 * architecture.
 * @tparam DataTypeA       Data type of matrix A
 * @tparam DataTypeB       Data type of matrix B
 * @tparam DataTypeAcc     Data type of the accumulator
 * @tparam BlockM          Block M size of the MMA operation
 * @tparam BlockN          Block N size of the MMA operation
 * @tparam TestBlockK      Block K size to start testing for support on the current target
 * architecture
 * @tparam GfxTargetId     Target architecture id
 * @tparam Enable          SFINAE enabler
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t TestBlockK  = 128u,
          uint32_t GfxTargetId = get_target_arch_id(),
          typename Enable      = void>
struct MmaDefaultSelector;

/*! @concept MmaSelectorI
 *  @brief  Expresses the required members for each MmaSelector class.
 *  @tparam MmaSelector The MmaSelector to be tested.
 */
template <typename MmaSelector>
concept MmaSelectorI = requires(MmaSelector op) {
    // Selectors should have a resulting SelectedOp type
    typename MmaSelector::SelectedOp;
};

} // namespace ck::tile::core::arch::mma
