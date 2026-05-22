// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: meta — operator structs, Op variant. No runtime, no CK deps.
//
// Operators are the edges of a Signature's compute graph. Each operator
// names its tensor slots as string_view labels (e.g., "A", "bias", "query")
// that refer to tensors declared elsewhere in the Signature. The Signature
// owns the tensor definitions; operators just reference them by name.
//
// This separation means operators are reusable across different tensor
// configurations — a GemmOp doesn't care whether its "lhs" is FP16 or BF16,
// Row or Col. That's resolved later when the Signature is validated.
//
// The Op variant is the closed set of supported operator types. Adding a
// new operator means adding a struct here and a variant alternative.
// Fused operations (like FMHA) are single operators — not chains of
// elementwise + GEMM — because CK Tile implements them as monolithic kernels.

#pragma once

#include <rocm_ck/datatype.hpp>

#include <string_view>
#include <variant>

namespace rocm_ck {

// Matrix multiplication: out = lhs x rhs.
// acc_dtype is the accumulation type — defaults to FP32, the universal safe
// choice across all input types.
struct GemmOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
    DataType acc_dtype = DataType::FP32;
};

// Element-wise addition: out = lhs + rhs.
struct AddOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
};

// Element-wise multiplication: out = lhs * rhs.
struct MulOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
};

// ReLU activation: out = max(0, in).
struct ReluOp
{
    std::string_view in;
    std::string_view out;
};

// Fast GELU approximation: out = in * sigmoid(1.702 * in).
struct FastGeluOp
{
    std::string_view in;
    std::string_view out;
};

// Exact GELU: out = 0.5 * in * (1 + erf(in / sqrt(2))).
struct GeluOp
{
    std::string_view in;
    std::string_view out;
};

// SiLU (Swish) activation: out = in * sigmoid(in).
struct SiluOp
{
    std::string_view in;
    std::string_view out;
};

// Sigmoid activation: out = 1 / (1 + exp(-in)).
struct SigmoidOp
{
    std::string_view in;
    std::string_view out;
};

// Softmax: out[i] = exp(in[i]) / sum(exp(in)), reduction along last dimension.
struct SoftmaxOp
{
    std::string_view in;
    std::string_view out;
};

// Scalar multiply: out = in * scale.
// 'scale' names a Scalar in the Signature, not a tensor.
struct ScaleOp
{
    std::string_view in;
    std::string_view out;
    std::string_view scale;
};

// The closed set of supported operators. std::monostate marks empty slots.
using Op = std::variant<std::monostate,
                        GemmOp,
                        AddOp,
                        MulOp,
                        ReluOp,
                        FastGeluOp,
                        GeluOp,
                        SiluOp,
                        SigmoidOp,
                        SoftmaxOp,
                        ScaleOp>;

} // namespace rocm_ck
