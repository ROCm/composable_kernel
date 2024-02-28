// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

namespace ck_tile {

enum struct address_space_enum
{
    generic,
    global,
    lds,
    sgpr,
    vgpr,
};

} // namespace ck_tile
