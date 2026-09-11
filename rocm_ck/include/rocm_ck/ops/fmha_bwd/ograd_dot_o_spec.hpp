// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compile-time configuration and named slot constants for the FMHA BWD
// OGradDotO kernel family.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval makeSpec() factory, and named slot
// constants. No runtime code, no HIP dependency.
//
// Compilation boundary:
//   _spec.hpp (this) -- consteval factory + slot constants (both passes)
//   _api.hpp         -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp         -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/ops/fmha_bwd/common.hpp>

#include <rocm_ck/datatype.hpp>

#include <cstdint>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Signature / Algorithm / Config / Kernel
// ---------------------------------------------------------------------------

/// Signature: describes WHAT the kernel computes (data types, dimensions, mode).
struct FmhaBwdOGradDotOSignature
{
    DataType dtype; // fp16 or bf16 (controls O, dO types; D is always float)
    int hdim_v;     // head dimension: 32, 64, 96, 128, 256
    FmhaMode mode;  // batch or group
};

/// Algorithm: describes HOW the kernel executes (padding, occupancy, block size).
struct FmhaBwdOGradDotOAlgorithm
{
    bool pad_seqlen_q;     // kPadSeqLenQ
    bool pad_hdim_v;       // kPadHeadDimV
    int block_per_cu = 2;  // occupancy hint (default from TileFmhaBwdOGradDotOTraits)
    int block_size   = 64; // tile_m0 = kM0 = kBlockSize
};

/// Config: user-facing Signature + Algorithm pair.
struct FmhaBwdOGradDotOConfig
{
    FmhaBwdOGradDotOSignature signature;
    FmhaBwdOGradDotOAlgorithm algorithm;
};

/// Validated kernel descriptor -- structural type, safe for use as NTTP.
/// All optional/default values are resolved; no std::optional.
struct FmhaBwdOGradDotOSpec
{
    DataType dtype;
    int hdim_v;
    FmhaMode mode;
    bool pad_seqlen_q;
    bool pad_hdim_v;
    int block_per_cu;
    int block_size;
};

/// Validate config and produce a structural kernel descriptor.
/// Overload resolution: each kernel family has its own Config type,
/// so makeSpec(FmhaBwdOGradDotOConfig) is unambiguous.
consteval FmhaBwdOGradDotOSpec makeSpec(FmhaBwdOGradDotOConfig cfg)
{
    auto sig  = cfg.signature;
    auto algo = cfg.algorithm;

    if(sig.dtype != DataType::FP16 && sig.dtype != DataType::BF16)
        throw "FmhaBwdOGradDotO only supports FP16 or BF16"
              " (O/dO types; D is always float)";

    // Valid head dimensions per BWD_DOT_DO_O_HDIMS in dispatcher
    if(sig.hdim_v != 32 && sig.hdim_v != 64 && sig.hdim_v != 96 && sig.hdim_v != 128 &&
       sig.hdim_v != 256)
        throw "hdim_v must be one of {32, 64, 96, 128, 256}";

    // BlockFmhaBwdOGradDotOPipelineProblem requires:
    //   0 < kBlockSize && kBlockSize % get_warp_size() == 0
    if(algo.block_size <= 0)
        throw "block_size must be positive";
    if(algo.block_size % 64 != 0)
        throw "block_size must be divisible by warp size (64)";

    if(algo.block_per_cu <= 0)
        throw "block_per_cu must be positive";

    // Group mode requires seqlen padding (variable-length sequences)
    if(sig.mode == FmhaMode::GROUP && !algo.pad_seqlen_q)
        throw "group mode requires pad_seqlen_q=true"
              " (variable-length sequences)";

    return {sig.dtype,
            sig.hdim_v,
            sig.mode,
            algo.pad_seqlen_q,
            algo.pad_hdim_v,
            algo.block_per_cu,
            algo.block_size};
}

// Compile canary: GROUP mode exercises pad_seqlen_q constraint.
// clang-format off
static_assert(makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}}).mode == FmhaMode::GROUP);
// clang-format on

// ---------------------------------------------------------------------------
// Named slot constants for generic Args
// ---------------------------------------------------------------------------

/// Named slot indices for FmhaBwdOGradDotO tensors and scalars within
/// rocm_ck::Args. Prevents off-by-one slot mapping errors.
namespace fmha_bwd_ograd_dot_o_slots {

// Tensor slots (always present)
constexpr int O  = 0; // const void*: output from forward pass
constexpr int DO = 1; // const void*: output gradient
constexpr int D  = 2; // void*: OGradDotO result (rowsum output)

// Tensor slots (group mode only)
constexpr int SEQSTART_Q = 3; // const int32_t*: sequence start offsets
constexpr int SEQLEN_Q   = 4; // const int32_t*: per-batch actual lengths

// Scalar slots
constexpr int P_UNDROP = 0; // float: 1 / (1 - dropout_rate)

/// Number of tensor slots required for a given kernel configuration.
constexpr int requiredTensors(FmhaBwdOGradDotOSpec k)
{
    return (k.mode == FmhaMode::GROUP) ? 5 : 3;
}

/// Number of scalar slots required (always 1: p_undrop).
constexpr int requiredScalars(FmhaBwdOGradDotOSpec /*k*/) { return 1; }

} // namespace fmha_bwd_ograd_dot_o_slots

} // namespace rocm_ck
