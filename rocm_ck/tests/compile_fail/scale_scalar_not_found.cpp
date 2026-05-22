// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: ScaleOp references scalar "alpha" but no Scalar entry exists.
// Expected error: "ScaleOp.scale references undeclared Scalar"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(
    Signature{.dtype = DataType::FP16, .ops = {ScaleOp{.in = "A", .out = "B", .scale = "alpha"}}});
