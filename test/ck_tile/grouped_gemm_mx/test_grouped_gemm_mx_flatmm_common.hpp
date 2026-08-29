// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_mx_flatmm_util.hpp"

using F8  = ck_tile::fp8_t;
using F16 = ck_tile::half_t;
using F32 = float;
using F4  = ck_tile::pk_fp4_t;
using F6  = ck_tile::pk_fp6x16_t;
