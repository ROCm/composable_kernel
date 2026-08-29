// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: GemmOp has empty tensor name in lhs slot.
// Expected error: "operator slot has empty tensor name"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad =
    resolve(Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "", .rhs = "B", .out = "C"}}});
