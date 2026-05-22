// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Tensor.quantize has empty scale_name.
// Expected error: "Tensor .quantize has empty scale_name"

#include <rocm_ck/resolve.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(
    Signature{.dtype   = DataType::FP16,
              .tensors = {Tensor{
                  .name = "B", .dtype = DataType::I4, .quantize = Quantization{.scale_name = ""}}},
              .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
