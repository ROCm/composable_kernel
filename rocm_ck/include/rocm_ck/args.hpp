// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: abi — shared between host and device. Trivially copyable, no CK deps.
//
// Args is a hardware buffer for passing data between CPU and GPU during a
// kernel call. It carries raw pointers, shapes, strides, and scalar values —
// nothing more. All semantic meaning (which tensor is "A", which scalar is
// "alpha", input vs output) lives in the Signature, not here.
//
// This is deliberately one type for all operations. Per-operation structs
// (GemmArgs, FmhaArgs, ...) would make the dispatcher a closed set — adding
// an operation means adding a type, updating launch code, and changing the
// kpack format. A generic buffer keeps the dispatcher open.
//
// Capacity limits (kMaxRank=6, kMaxTensors=16, kMaxScalars=16) are sized to
// the most demanding current operation (FMHA backward: ~12 tensors, ~12
// scalars, rank-6 for grouped 3D conv). If a future operation exceeds these,
// bump the constants — the layout is not versioned, and the 4KB HSA kernarg
// budget has room. Don't over-provision speculatively.
//
// Key constraints:
//   - Trivially copyable, standard layout — required for HSA kernarg passing.
//   - Fixed-capacity arrays, no heap — sizeof fits the 4KB kernarg budget.
//   - const void* for all tensor pointers — the entry kernel casts to the
//     concrete type. Input vs output semantics live in the Signature.
//   - No runtime type tags on scalars — the Signature declares types at
//     compile time. The entry kernel reads the correct union member.
//   - Slot ordering is the invariant: tensors[i] maps to Signature::tensors[i].

#pragma once

#include <rocm_ck/index_t.hpp>

#include <array>
#include <cstdint>

namespace rocm_ck {

// When changing these, update the byte-size comments on TensorArg and Args fields.
constexpr int kMaxRank    = 6;  // grouped 3D conv = GNCDHW = rank 6
constexpr int kMaxTensors = 16; // FMHA backward uses ~12
constexpr int kMaxScalars = 16; // FMHA with masking+dropout needs ~12

struct TensorArg
{
    const void* ptr;                            //  8 bytes  (offset 0)
    std::array<index_t, kMaxRank> lengths;      // 24 bytes  (offset 8)   — int32
    std::array<long_index_t, kMaxRank> strides; // 48 bytes  (offset 32)  — int64
};

// FP16/BF16/FP8 scalars use f32 — scalar precision >= tensor precision.
union ScalarValue
{
    float f32;
    int32_t i32;
    uint32_t u32;
    double f64;
    int64_t i64;
    uint64_t u64;
};

// Slot ordering matches Signature: tensors[i] <-> Signature::tensors[i].
struct Args
{
    std::array<TensorArg, kMaxTensors> tensors;   // 16 x 80 = 1280 bytes
    std::array<ScalarValue, kMaxScalars> scalars; // 16 x  8 =  128 bytes

    index_t batch_count                                 = 0;       //  4 bytes
    std::array<long_index_t, kMaxTensors> batch_strides = {};      // 16 x 8 = 128 bytes
    void* workspace_ptr                                 = nullptr; //  8 bytes
};

constexpr std::array<index_t, kMaxRank> makeShape(
    index_t d0, index_t d1 = 0, index_t d2 = 0, index_t d3 = 0, index_t d4 = 0, index_t d5 = 0)
{
    return {d0, d1, d2, d3, d4, d5};
}

constexpr std::array<long_index_t, kMaxRank> makeStrides(long_index_t s0,
                                                         long_index_t s1 = 0,
                                                         long_index_t s2 = 0,
                                                         long_index_t s3 = 0,
                                                         long_index_t s4 = 0,
                                                         long_index_t s5 = 0)
{
    return {s0, s1, s2, s3, s4, s5};
}

} // namespace rocm_ck
