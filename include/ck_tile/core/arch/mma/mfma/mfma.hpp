// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../arch.hpp"
#include "../mma_common.hpp"

namespace ck_tile::core::arch::mma {
/*! @struct MfmaOp
 * @brief Meta-tag for the MFMA operation. This will be used in the MmaOp struct to
 * identify the operation as an MFMA operation.
 */
struct MfmaOp;

} // namespace ck_tile::core::arch::mma

// Include the architecture-specific MFMA implementations and traits
#include "mfma_gfx9.hpp"
#include "mfma_traits.hpp"
#include "mfma_selector.hpp"
