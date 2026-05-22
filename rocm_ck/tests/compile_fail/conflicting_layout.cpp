// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: two GemmOps imply conflicting layouts for the same tensor.
//
// GemmOp1 outputs "C" → implied Row layout.
// GemmOp2 uses "C" as rhs → implied Col layout.
// These conflict: "C" can't be both Row and Col.
//
// Expected error: "conflicting layout for tensor"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.dtype = DataType::FP16,
                                       .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                 GemmOp{.lhs = "X", .rhs = "C", .out = "Y"}}});
