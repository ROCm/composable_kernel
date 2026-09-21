// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// FMHA-only dispatcher header. Does not pull in GEMM or Conv types.

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/fmha_types.hpp"
#include "ck_tile/dispatcher/fmha_problem.hpp"
#include "ck_tile/dispatcher/fmha_kernel_key.hpp"
#include "ck_tile/dispatcher/fmha_kernel_instance.hpp"
#include "ck_tile/dispatcher/fmha_registry.hpp"
#include "ck_tile/dispatcher/fmha_dispatcher.hpp"
#include "ck_tile/dispatcher/fmha_kernel_decl.hpp"
#include "ck_tile/dispatcher/backends/generated_fmha_backend.hpp"
