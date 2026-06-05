// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/numeric/tfloat32.hpp"

using INT8  = ck_tile::int8_t;
using INT32 = ck_tile::int32_t;

using F16 = ck_tile::half_t;
using F32 = float;
using F8  = ck_tile::fp8_t;
using F4  = ck_tile::pk_fp4_t;

using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;

using I4 = ck_tile::pk_int4_t;

using TF32 = ck_tile::tf32_t;
