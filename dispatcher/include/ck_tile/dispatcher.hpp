// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Full dispatcher header - includes ALL operation types.
/// For minimal includes, use the per-operation headers instead:
///   ck_tile/dispatcher_gemm.hpp      -- GEMM only
///   ck_tile/dispatcher_conv.hpp      -- Grouped Convolution only
///   ck_tile/dispatcher_fmha.hpp      -- FMHA only

// Core (needed by all ops)
#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

// GEMM
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/kernel_config.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/dispatcher/utils.hpp"

// Grouped Convolution
#include "ck_tile/dispatcher/grouped_conv_config.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/grouped_conv_utils.hpp"

// FMHA
#include "ck_tile/dispatcher/fmha_types.hpp"
#include "ck_tile/dispatcher/fmha_problem.hpp"
#include "ck_tile/dispatcher/fmha_kernel_key.hpp"
#include "ck_tile/dispatcher/fmha_kernel_instance.hpp"
#include "ck_tile/dispatcher/fmha_registry.hpp"
#include "ck_tile/dispatcher/fmha_dispatcher.hpp"
#include "ck_tile/dispatcher/fmha_kernel_decl.hpp"
#include "ck_tile/dispatcher/backends/generated_fmha_backend.hpp"
