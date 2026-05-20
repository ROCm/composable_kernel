// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — GpuTarget enum. No runtime, no CK deps.

#pragma once

#include <cstdint>

namespace rocm_ck {

// ISA target identifiers (matching -mcpu flags), not marketing names.
enum class GpuTarget : uint8_t
{
    gfx90a,  // CDNA 2
    gfx942,  // CDNA 3
    gfx950,  // CDNA 4
    gfx1100, // RDNA 3
    gfx1101, // RDNA 3
    gfx1102, // RDNA 3
    gfx1150, // RDNA 3.5
    gfx1151, // RDNA 3.5
};

} // namespace rocm_ck
