// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: ScaleOp references undeclared scalar.
// Expected error: "ScaleOp.scale references undeclared Scalar"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{
    .dtype = DataType::FP16, .ops = {ScaleOp{.in = "X", .out = "Y", .scale = "missing"}}});
