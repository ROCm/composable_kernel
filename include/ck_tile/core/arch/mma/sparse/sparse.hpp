// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

} // namespace ck_tile::core::arch::mma

// Include sparse MFMA traits and architecture-specific implementations
#include "ck_tile/core/arch/mma/sparse/mfma/sparse_gfx9.hpp"
#include "ck_tile/core/arch/mma/sparse/wmma/sparse_gfx12.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_selector.hpp"
