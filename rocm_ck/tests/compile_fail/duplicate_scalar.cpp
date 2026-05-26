// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: duplicate scalar names.
// Expected error: "duplicate scalar name in Signature"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad =
    resolve(Signature{.dtype   = DataType::FP16,
                      .scalars = {Scalar{.name = "alpha"}, Scalar{.name = "alpha"}},
                      .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
