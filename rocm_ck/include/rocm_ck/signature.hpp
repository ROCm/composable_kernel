// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: meta — Signature, Tensor, Scalar. No runtime, no CK deps.
//
// Signature: the complete description of WHAT a kernel computes.
//
// A directed compute graph where tensors are nodes and operators are edges.
// Operators reference tensors by name (SSA-style). Shared names between
// operator outputs and inputs form graph edges.
//
// This header has NO CK Tile dependency. It is included by both host code
// and device code to share the signature definition.

#pragma once

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/ops.hpp>

#include <array>
#include <optional>
#include <string_view>

namespace rocm_ck {

/// Quantization metadata for a tensor stored in a packed/quantized format.
///
/// Describes how a quantized tensor (e.g., INT4 weights) is encoded.
/// The scale tensor is a separate physical tensor with per-group factors;
/// group_size is the number of consecutive elements sharing one scale along K.
struct Quantization
{
    std::string_view scale_name;           // name of the scale tensor (e.g., "scale")
    DataType scale_dtype = DataType::FP32; // element type of the scale tensor
    int group_size       = 128;            // elements per quantization group
};

/// A tensor node in the signature's compute graph.
///
/// Default values mean "inherit from the operator slot":
///   dtype    = nullopt  -> inherit from Signature::dtype cascade
///   rank     = 0        -> inherit from operator (e.g., GemmOp implies rank 2)
///   layout   = Auto     -> inherit from operator (e.g., GemmOp::lhs implies Row)
///   quantize = nullopt  -> not quantized
struct Tensor
{
    std::string_view name;
    std::optional<DataType> dtype        = std::nullopt;
    int rank                             = 0;
    Layout layout                        = Layout::Auto;
    std::optional<Quantization> quantize = std::nullopt;
};

/// A named scalar parameter (e.g., alpha, beta, scale).
struct Scalar
{
    std::string_view name;
    DataType dtype = DataType::FP32;
};

/// The complete description of WHAT a kernel computes.
///
/// A directed compute graph where tensors are nodes and operators are edges.
/// Each operator output gets a unique name; shared names form graph edges.
///
/// Example — simple fp16 GEMM:
///   {.dtype = FP16, .ops = {GemmOp{.lhs="A", .rhs="B", .out="C"}}}
///
/// Example — GEMM + bias + ReLU:
///   {.dtype = FP16,
///    .ops = {GemmOp{.lhs="A", .rhs="B", .out="C"},
///            AddOp{.lhs="C", .rhs="bias", .out="D"},
///            ReluOp{.in="D", .out="E"}}}
// kMaxTensors and kMaxScalars are defined in args.hpp (canonical source).
// kMaxOps is Signature-specific (operators, not kernel arguments).
constexpr int kMaxOps = 8;

struct Signature
{
    std::optional<DataType> dtype;
    std::array<Tensor, kMaxTensors> tensors = {};
    std::array<Scalar, kMaxScalars> scalars = {};
    std::array<Op, kMaxOps> ops             = {};
};

} // namespace rocm_ck
