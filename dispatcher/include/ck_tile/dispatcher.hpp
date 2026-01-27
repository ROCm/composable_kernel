// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Main dispatcher header - includes all core components
/// Use this for convenient access to the full dispatcher API

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/kernel_config.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/dispatcher/utils.hpp"
