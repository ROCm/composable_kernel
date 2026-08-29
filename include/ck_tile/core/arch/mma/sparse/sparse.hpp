// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Include sparse MFMA traits and architecture-specific implementations
#include "ck_tile/core/arch/mma/sparse/mfma/sparse_gfx9.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_selector.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/arch/mma/sparse/wmma/sparse_gfx12.hpp"
