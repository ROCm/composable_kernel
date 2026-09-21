// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: no dtype set on tensors or signature.
// Expected error: "tensor dtype unresolvable"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
