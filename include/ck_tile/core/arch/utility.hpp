// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

namespace ck_tile {

// TODO: we have "memory" clobber here because this inline asm is used for async copy
CK_TILE_DEVICE void m0_set_with_memory(index_t v)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

// NOTE: this is an immediate value
CK_TILE_DEVICE void m0_inc_with_memory(index_t v)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
}

} // namespace ck_tile
