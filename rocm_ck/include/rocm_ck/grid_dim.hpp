// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HIP-free grid dimension type for host-only kernel launch calculations.

#pragma once

namespace rocm_ck {

/// HIP-free replacement for dim3. Consumers convert to dim3 at the HIP
/// call site: `dim3 grid(g.x, g.y, g.z)`.
struct GridDim
{
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

} // namespace rocm_ck
