// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: meta -- consteval resolve(), C++20 concepts. No runtime, no CK deps.
//
// Signature resolution: resolves a Signature into concrete tensor descriptors.
//
// resolve() walks the operator graph, collects tensor slots with operator-implied
// defaults, propagates rank/layout through connected tensors, merges explicit
// Tensor entries, and applies the dtype cascade. All at compile time (consteval).
//
// Op dispatch uses C++20 concepts to classify ops by structural shape:
//   BinaryOpLike  -- has {lhs, rhs, out} string_view members
//   UnaryOpLike   -- has {in, out} string_view members
// A single visitOp() function is the only place the Op variant type list
// appears. Adding a new op requires one line in visitOp(); concepts handle
// generic registration and propagation automatically.
//
// This header has NO CK Tile dependency.

#pragma once

#include <rocm_ck/signature.hpp>
#include <rocm_ck/resolved_tensor.hpp>

#include <array>
#include <type_traits>

namespace rocm_ck {

// ============================================================================
// Op structural concepts -- classify ops by their tensor slot shape
// ============================================================================

/// Ops with three tensor slots: lhs, rhs, out (e.g., AddOp, MulOp).
/// Excludes ops that also have {in} to prevent overlap with UnaryOpLike.
template <typename T>
concept BinaryOpLike = requires(const T& t) {
    t.lhs;
    t.rhs;
    t.out;
} && !requires(const T& t) { t.in; };

/// Ops with two tensor slots: in, out (e.g., ReluOp, SigmoidOp, ScaleOp).
/// Excludes ops that also have {lhs, rhs} to prevent overlap with BinaryOpLike.
template <typename T>
concept UnaryOpLike = requires(const T& t) {
    t.in;
    t.out;
} && !requires(const T& t) {
    t.lhs;
    t.rhs;
};

// ============================================================================
// visitOp -- single consteval dispatch point for all Op types
//
// Adding a new Op alternative requires no changes here -- std::visit
// enforces exhaustiveness at compile time via the visitor's operator().
// ============================================================================

template <typename F>
consteval auto visitOp(const Op& op, F&& f)
{
    return std::visit(std::forward<F>(f), op);
}

// ============================================================================
// ResolvedSignature
// ============================================================================

/// Resolved signature: all tensors and scalars have concrete metadata.
/// Rank and layout are concrete for tensors with operator-implied defaults
/// or explicit Tensor entries; may remain 0/Auto for tensors where the
/// operation type does not specify them (e.g., standalone AddOp).
struct ResolvedSignature
{
    int num_tensors                                 = 0;
    std::array<ResolvedTensor, kMaxTensors> tensors = {};

    int num_scalars                         = 0;
    std::array<Scalar, kMaxScalars> scalars = {};

    /// Find a resolved tensor by name. Compile-time error if not found.
    consteval ResolvedTensor tensor(std::string_view name) const
    {
        for(int i = 0; i < num_tensors; ++i)
            if(tensors[i].name == name)
                return tensors[i];
        throw "tensor() -- name not found in resolved signature; "
              "check that it appears in an op slot or Tensor entry";
    }

    /// Find a resolved tensor's slot index by name. Compile-time error if not found.
    consteval int tensorIndex(std::string_view name) const
    {
        for(int i = 0; i < num_tensors; ++i)
            if(tensors[i].name == name)
                return i;
        throw "tensorIndex() -- name not found in resolved signature; "
              "check that it appears in an op slot or Tensor entry";
    }

    /// Find a resolved scalar by name. Compile-time error if not found.
    consteval Scalar scalar(std::string_view name) const
    {
        for(int i = 0; i < num_scalars; ++i)
            if(scalars[i].name == name)
                return scalars[i];
        throw "scalar() -- name not found; "
              "add a Scalar entry with this name to the Signature";
    }

    /// Find a resolved scalar's slot index by name. Compile-time error if not found.
    consteval int scalarIndex(std::string_view name) const
    {
        for(int i = 0; i < num_scalars; ++i)
            if(scalars[i].name == name)
                return i;
        throw "scalarIndex() -- name not found; "
              "add a Scalar entry with this name to the Signature";
    }

    /// Find a tensor slot index by name. Returns -1 if not found.
    /// Unlike tensorIndex() (consteval), callable at both compile time and runtime.
    constexpr int findTensor(std::string_view name) const
    {
        for(int i = 0; i < num_tensors; ++i)
            if(tensors[i].name == name)
                return i;
        return -1;
    }

    /// Find a scalar slot index by name. Returns -1 if not found.
    /// Unlike scalarIndex() (consteval), callable at both compile time and runtime.
    constexpr int findScalar(std::string_view name) const
    {
        for(int i = 0; i < num_scalars; ++i)
            if(scalars[i].name == name)
                return i;
        return -1;
    }
};

// ============================================================================
// Op slot visitors -- concept-driven, generic handling for binary/unary ops
// ============================================================================

/// Register all tensor slots of an op. Returns the output tensor name.
///
/// Uses concepts for generic dispatch:
///   GemmOp  -- special case: sets operator-implied rank/layout defaults
///   ScaleOp -- special case: validates scalar reference against sig.scalars[]
///   BinaryOpLike -- generic: registers lhs, rhs, out
///   UnaryOpLike  -- generic: registers in, out
///
/// Adding a new BinaryOpLike or UnaryOpLike op requires no changes here.
consteval std::string_view collectTensorSlotsFromOp(const Op& op,
                                                    const Signature& sig,
                                                    auto& find_or_add,
                                                    auto& set_if_unknown)
{
    return visitOp(op, [&](const auto& typed_op) -> std::string_view {
        using T = std::remove_cvref_t<decltype(typed_op)>;

        if constexpr(std::is_same_v<T, std::monostate>)
        {
            return {};
        }
        else if constexpr(std::is_same_v<T, GemmOp>)
        {
            // GemmOp has operator-implied rank/layout defaults
            set_if_unknown(find_or_add(typed_op.lhs), 2, Layout::Row);
            set_if_unknown(find_or_add(typed_op.rhs), 2, Layout::Col);
            set_if_unknown(find_or_add(typed_op.out), 2, Layout::Row);
            return typed_op.out;
        }
        else if constexpr(std::is_same_v<T, ScaleOp>)
        {
            // ScaleOp is unary + scalar reference validation
            find_or_add(typed_op.in);
            find_or_add(typed_op.out);
            if(typed_op.scale.empty())
                throw "ScaleOp.scale must name a Scalar parameter";
            bool found_scalar = false;
            for(int si = 0; si < kMaxScalars; ++si)
            {
                if(!sig.scalars[si].name.empty() && sig.scalars[si].name == typed_op.scale)
                {
                    found_scalar = true;
                    break;
                }
            }
            if(!found_scalar)
                throw "ScaleOp.scale references undeclared Scalar -- "
                      "add a matching Scalar entry to the Signature";
            return typed_op.out;
        }
        else if constexpr(BinaryOpLike<T>)
        {
            find_or_add(typed_op.lhs);
            find_or_add(typed_op.rhs);
            find_or_add(typed_op.out);
            return typed_op.out;
        }
        else if constexpr(UnaryOpLike<T>)
        {
            find_or_add(typed_op.in);
            find_or_add(typed_op.out);
            return typed_op.out;
        }
        else
        {
            throw "unhandled Op type in collectTensorSlotsFromOp -- "
                  "add explicit handling or satisfy BinaryOpLike/UnaryOpLike";
        }
    });
}

/// Propagate rank/layout through an op's connected tensor slots.
///
/// Binary ops propagate from the first slot with known rank/layout to all others.
/// Unary ops propagate between in and out.
/// GemmOp is skipped (rank/layout already set by collectTensorSlotsFromOp).
///
/// Adding a new BinaryOpLike or UnaryOpLike op requires no changes here.
consteval void propagateRankLayout(const Op& op, auto& propagate_binary, auto& propagate_unary)
{
    visitOp(op, [&](const auto& typed_op) {
        using T = std::remove_cvref_t<decltype(typed_op)>;

        if constexpr(std::is_same_v<T, std::monostate> || std::is_same_v<T, GemmOp>)
        {
            // monostate: nothing to do
            // GemmOp: rank/layout already set in collectTensorSlotsFromOp
        }
        else if constexpr(BinaryOpLike<T>)
        {
            propagate_binary(typed_op.lhs, typed_op.rhs, typed_op.out);
        }
        else if constexpr(UnaryOpLike<T>)
        {
            propagate_unary(typed_op.in, typed_op.out);
        }
        else
        {
            throw "unhandled Op type in propagateRankLayout -- "
                  "add explicit handling or satisfy BinaryOpLike/UnaryOpLike";
        }
    });
}

// ============================================================================
// resolve()
// ============================================================================

/// Resolve a Signature into concrete tensor and scalar descriptors.
///
/// Phases:
///   1. Register tensor slots from operators (with op-implied rank/layout)
///   2. Propagate rank/layout through connected tensors (forward + backward)
///   3. Merge explicit Tensor entries from sig.tensors (overrides propagation)
///   4. Apply dtype cascade: explicit tensor -> sig.dtype -> error
///   5. Collect declared scalars, build result
///
/// GemmOp slots get operator-implied defaults:
///   lhs -> rank 2, Row;  rhs -> rank 2, Col;  out -> rank 2, Row
///
/// Binary ops (AddOp, MulOp) and unary ops propagate rank/layout from
/// the first connected slot that has known values.
consteval ResolvedSignature resolve(const Signature& sig)
{
    // --- Intermediate tracking ---
    struct Info
    {
        std::string_view name;
        bool dtype_set    = false;
        DataType dtype    = DataType::FP32;
        int rank          = 0;
        Layout layout     = Layout::Auto;
        bool has_quantize = false;
        Quantization quantize_info{}; // only valid when has_quantize == true
    };

    Info infos[kMaxTensors] = {};
    int num                 = 0;

    // Find tensor by name, return index or -1.
    auto find = [&](std::string_view name) -> int {
        for(int i = 0; i < num; ++i)
            if(infos[i].name == name)
                return i;
        return -1;
    };

    // Find tensor by name; add if not present.
    auto find_or_add = [&](std::string_view name) -> int {
        if(name.empty())
            throw "operator slot has empty tensor name";
        int idx = find(name);
        if(idx >= 0)
            return idx;
        if(num >= kMaxTensors)
            throw "too many unique tensors in Signature (max 16)";
        infos[num].name = name;
        return num++;
    };

    // Set rank/layout only if currently unknown. Error on conflicting values.
    // Returns true if anything changed.
    auto set_if_unknown = [&](int idx, int rank, Layout layout) -> bool {
        bool changed = false;
        if(rank != 0)
        {
            if(infos[idx].rank == 0)
            {
                infos[idx].rank = rank;
                changed         = true;
            }
            else if(infos[idx].rank != rank)
                throw "conflicting rank for tensor -- two operators imply different ranks; "
                      "check that shared tensor names are intentional";
        }
        if(layout != Layout::Auto)
        {
            if(infos[idx].layout == Layout::Auto)
            {
                infos[idx].layout = layout;
                changed           = true;
            }
            else if(infos[idx].layout != layout)
                throw "conflicting layout for tensor -- two operators imply different layouts; "
                      "check that shared tensor names are intentional";
        }
        return changed;
    };

    // ================================================================
    // Phase 1: Register tensor slots from operators
    // ================================================================
    std::string_view output_names[kMaxOps] = {};
    int num_outputs                        = 0;

    for(int i = 0; i < kMaxOps; ++i)
    {
        std::string_view out_name =
            collectTensorSlotsFromOp(sig.ops[i], sig, find_or_add, set_if_unknown);

        // SSA uniqueness: each output name may appear at most once
        if(!out_name.empty())
        {
            for(int j = 0; j < num_outputs; ++j)
                if(output_names[j] == out_name)
                    throw "tensor name produced by multiple operators "
                          "(each output name must be unique)";
            output_names[num_outputs++] = out_name;
        }
    }

    // ================================================================
    // Phase 2: Propagate rank/layout through connected tensors
    // ================================================================

    bool changed = false;

    auto propagate_binary =
        [&](std::string_view lhs_name, std::string_view rhs_name, std::string_view out_name) {
            int li = find(lhs_name);
            int ri = find(rhs_name);
            int oi = find(out_name);
            if(li < 0 || ri < 0 || oi < 0)
                return;

            int src = -1;
            if(infos[li].rank != 0)
                src = li;
            else if(infos[ri].rank != 0)
                src = ri;
            else if(infos[oi].rank != 0)
                src = oi;

            if(src >= 0)
            {
                changed |= set_if_unknown(li, infos[src].rank, infos[src].layout);
                changed |= set_if_unknown(ri, infos[src].rank, infos[src].layout);
                changed |= set_if_unknown(oi, infos[src].rank, infos[src].layout);
            }
        };

    auto propagate_unary = [&](std::string_view in_name, std::string_view out_name) {
        int ii = find(in_name);
        int oi = find(out_name);
        if(ii < 0 || oi < 0)
            return;

        if(infos[ii].rank != 0)
            changed |= set_if_unknown(oi, infos[ii].rank, infos[ii].layout);
        else if(infos[oi].rank != 0)
            changed |= set_if_unknown(ii, infos[oi].rank, infos[oi].layout);
    };

    // Propagate until stable. Chains converge in 1 pass, diamonds in 2-3.
    // Cap at 4 to bound compile time. Unreachable with kMaxOps=8 (a graph
    // requiring >4 passes would need more ops than fit), but guards against
    // future kMaxOps increases.
    constexpr int kMaxPropagationPasses = 4;
    for(int pass = 0; pass < kMaxPropagationPasses; ++pass)
    {
        changed = false;
        for(int i = 0; i < kMaxOps; ++i)
            propagateRankLayout(sig.ops[i], propagate_binary, propagate_unary);
        for(int i = kMaxOps - 1; i >= 0; --i)
            propagateRankLayout(sig.ops[i], propagate_binary, propagate_unary);
        if(!changed)
            break;
    }
    if(changed)
        throw "could not infer rank/layout for all tensors -- "
              "set rank and layout explicitly on Tensor entries, "
              "or reduce chained operations";

    // ================================================================
    // Phase 3: Merge explicit Tensor entries (override propagation)
    // ================================================================
    for(int i = 0; i < kMaxTensors; ++i)
    {
        if(sig.tensors[i].name.empty())
        {
            // Catch entries with metadata but no name (likely a mistake)
            if(sig.tensors[i].dtype.has_value() || sig.tensors[i].rank != 0 ||
               sig.tensors[i].layout != Layout::Auto || sig.tensors[i].quantize.has_value())
                throw "Tensor entry has metadata but no name";
            continue;
        }
        int idx = find_or_add(sig.tensors[i].name);
        if(sig.tensors[i].dtype.has_value())
        {
            infos[idx].dtype_set = true;
            infos[idx].dtype     = *sig.tensors[i].dtype;
        }
        if(sig.tensors[i].rank != 0)
            infos[idx].rank = sig.tensors[i].rank;
        if(sig.tensors[i].layout != Layout::Auto)
            infos[idx].layout = sig.tensors[i].layout;
        if(sig.tensors[i].quantize.has_value())
        {
            const auto& q = *sig.tensors[i].quantize;
            if(q.scale_name.empty())
                throw "Tensor .quantize has empty scale_name";
            // Auto-register the scale tensor
            int scale_idx = find_or_add(q.scale_name);
            if(infos[scale_idx].dtype_set && infos[scale_idx].dtype != q.scale_dtype)
                throw "tensor dtype conflicts with quantization scale_dtype "
                      "for the same name";
            infos[scale_idx].dtype_set = true;
            infos[scale_idx].dtype     = q.scale_dtype;
            set_if_unknown(scale_idx, 2, Layout::Row);
            // Track quantization on this tensor
            infos[idx].has_quantize  = true;
            infos[idx].quantize_info = q;
        }
    }

    // ================================================================
    // Phase 4: dtype cascade
    // ================================================================
    for(int i = 0; i < num; ++i)
    {
        if(!infos[i].dtype_set)
        {
            if(sig.dtype.has_value())
                infos[i].dtype = *sig.dtype;
            else
                throw "tensor dtype unresolvable: set tensor dtype or signature dtype";
        }
    }

    // ================================================================
    // Phase 5: Build result
    // ================================================================
    ResolvedSignature result{};
    result.num_tensors = num;
    for(int i = 0; i < num; ++i)
    {
        result.tensors[i] =
            ResolvedTensor{infos[i].name, infos[i].dtype, infos[i].rank, infos[i].layout};
        if(infos[i].has_quantize)
            result.tensors[i].quantize = ResolvedQuantization{infos[i].quantize_info.scale_name,
                                                              infos[i].quantize_info.scale_dtype,
                                                              infos[i].quantize_info.group_size};
    }

    // Collect declared scalars (pass-through -- no inference needed)
    for(int i = 0; i < kMaxScalars; ++i)
    {
        if(sig.scalars[i].name.empty())
            continue;
        // Check for duplicate scalar names
        for(int j = 0; j < result.num_scalars; ++j)
            if(result.scalars[j].name == sig.scalars[i].name)
                throw "duplicate scalar name in Signature";
        result.scalars[result.num_scalars++] = sig.scalars[i];
    }

    return result;
}

} // namespace rocm_ck
