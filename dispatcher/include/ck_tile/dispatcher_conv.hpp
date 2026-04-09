// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Grouped Convolution-only dispatcher header -- minimal include for conv operations.

#pragma once

// Core (needed by all ops)
#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

// Grouped Convolution
#include "ck_tile/dispatcher/grouped_conv_config.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
