// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — ResolvedTensor, ResolvedQuantization. No runtime, no CK deps.
//
// ResolvedTensor is the intermediate result of consteval resolution. It exists
// only at compile time — produced by Signature::resolve() and consumed by
// makeSpec(), both consteval. It never appears in compiled code.
//
// In the user-facing Signature, tensors can have Layout::Auto (inherit from
// operator slot) and omit fields with sensible defaults. After resolution,
// every field is concrete. The base fields (name, dtype, rank, layout)
// describe a plain dense tensor — enough for most operands (GEMM inputs,
// outputs, bias vectors). Some tensors carry additional metadata beyond the
// dense description. Block-quantized tensors (e.g., INT4 weights) need a
// scale tensor and group size. We use optional sub-structs for these
// extensions, keeping the common case clean without bloating every instance.
//
// Why std::string_view instead of FixedString?
//   ResolvedTensor is consteval-only — produced and consumed entirely at
//   compile time. No library loading, no runtime lifetime concerns. The
//   string_views point to string literals from user code (e.g.,
//   GemmOp{.lhs = "A"}), which have static storage duration — no dangling.
//   FixedString is required for PhysicalTensor because it IS used as a
//   template parameter (NTTP), which requires structural types (no pointers).
//   ResolvedTensor is never a template parameter.
//
// Plain aggregate — no methods, no validation. Resolution validates; this
// type just carries the result to makeSpec().

#pragma once

#include <rocm_ck/datatype.hpp>
#include <rocm_ck/layout.hpp>

#include <optional>
#include <string_view>

namespace rocm_ck {

// Present when a tensor carries block-quantized data (e.g., INT4 weights).
// The scale tensor is a separate entry in the Signature; this struct ties
// the quantized tensor to its scale.
struct ResolvedQuantization
{
    std::string_view scale_name;
    DataType scale_dtype;
    int group_size; // elements per quantization group
};

struct ResolvedTensor
{
    std::string_view name;
    DataType dtype;
    int rank                                     = 2;
    Layout layout                                = Layout::Row;
    std::optional<ResolvedQuantization> quantize = std::nullopt;
};

} // namespace rocm_ck
