// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "test_mx_flatmm_base.hpp"
#include "mx_flatmm_arch_traits.hpp"

// Convenience type aliases for use in test .cpp files
using FP4  = ck_tile::pk_fp4_t;
using FP6  = ck_tile::pk_fp6x16_t;
using FP8  = ck_tile::fp8_t;
using FP16 = ck_tile::fp16_t;

// Concrete test fixture - inherits all logic from TestMXFlatmmBase.
// Tuple layout: <ADataType, BDataType, CDataType, MXFlatmmArchTraits>
template <typename Tuple>
class TestMXFlatmm : public TestMXFlatmmBase<Tuple>
{
};
