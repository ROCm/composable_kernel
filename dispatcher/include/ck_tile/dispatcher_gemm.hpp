// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// GEMM-only dispatcher header. Does not pull in Conv or FMHA types.

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
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
/// GEMM-only dispatcher header -- minimal include for GEMM operations.

#pragma once

// Core (needed by all ops)
#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/example_args.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "ck_tile/dispatcher/utils.hpp"
