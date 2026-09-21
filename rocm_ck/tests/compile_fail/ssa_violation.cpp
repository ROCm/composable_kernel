// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: two GemmOps produce the same output tensor "C".
// Expected error: "tensor name produced by multiple operators"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.dtype = DataType::FP16,
                                       .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                 GemmOp{.lhs = "X", .rhs = "Y", .out = "C"}}});
