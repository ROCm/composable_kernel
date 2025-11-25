// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

/// Main dispatcher header - includes all core components
/// Use this for convenient access to the full dispatcher API

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"

// Optional: Kernel caching (include explicitly if needed)
// #include "ck_tile/dispatcher/kernel_cache.hpp"
