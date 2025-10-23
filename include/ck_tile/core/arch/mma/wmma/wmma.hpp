// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "../mma_common.hpp"

namespace ck_tile::core::arch::mma {
/*! @struct WmmaOp
 * @brief Meta-tag for the WMMA operation. This will be used in the MmaOp struct to
 * identify the operation as an WMMA operation.
 */
struct WmmaOp;

/*! @struct WmmaCtrlFlags
 * @brief Common wmma control flags for gfx11 and gfx12
 */
enum struct WmmaCtrlFlags : bool
{
    // Only has an effect on gfx11 when the accumulator is 16-bit
    // Determines which half of the 32-bit accum register to use
    // Low = bits [15:0]
    // High = bits[31:16]
    LOW  = false,
    HIGH = true,

    // Only has an effect on gfx11 / 12 when the input is 8-bit int
    // Signage indicator of inputs / accum
    UNSIGNED = false,
    SIGNED   = true
};

} // namespace ck_tile::core::arch::mma

// Include the architecture-specific WMMA implementations and traits
#include "wmma_gfx11.hpp"
#include "wmma_gfx12.hpp"
#include "wmma_selector.hpp"
#include "wmma_traits.hpp"
#include "wmma_transforms.hpp"
