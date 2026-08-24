// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/scale/mfma/selector.hpp"
#include "ck_tile/core/arch/mma/scale/wmma/selector.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @brief Selects the dense MmaOp used when a scale MmaOp is called without scale factors.
 *
 * By default, the dense operation has the same datatypes, shape and target as the scale operation.
 *
 * @tparam MmaOp The scale MmaOp to find a dense fallback for.
 */
template <typename MmaOp>
struct MmaOpDenseFallbackSelector
{
    using DenseOp = typename MmaDefaultSelector<typename MmaOp::ADataType,
                                                typename MmaOp::BDataType,
                                                typename MmaOp::CDataType,
                                                MmaOp::kM,
                                                MmaOp::kN,
                                                MmaOp::kK,
                                                typename MmaOpTraits<MmaOp>::CompilerTarget,
                                                MmaOpFamily::DENSE>::SelectedOp;
};

} // namespace ck_tile::core::arch::mma
