// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD ConvertDQ. Maps the validated kernel
// descriptor (FmhaBwdConvertDQSpec) to the CK Tile template chain.
//
// ConvertDQ sums split-K partial results from the deterministic dQ/dK/dV
// kernel and type-converts dQ_acc (fp32) to dQ (fp16/bf16) in one pass.
//
// Unpacks generic rocm_ck::Args via named slot constants, then constructs
// CK Tile's Kargs via aggregate initialization. No __builtin_bit_cast,
// no ABI matching required.
//
// Uses C++20 struct NTTPs: template <FmhaBwdConvertDQSpec K>.
//
// Compilation boundary:
//   _spec.hpp -- consteval factory + slot constants (both passes)
//   _api.hpp  -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp (this) -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "convert_dq_dev.hpp requires device compilation." \
       " Host code should include <rocm_ck/ops/fmha_bwd/convert_dq_api.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>

namespace rocm_ck {

/// Maps a FmhaBwdConvertDQSpec descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdConvertQGradKernel<Pipeline>
///     -> BlockFmhaBwdConvertQGrad<PipelineProblem>
///       -> BlockFmhaBwdConvertQGradPipelineProblem<AccDataType, QGradDataType,
///              BlockSize, kM0, kN0, kQKHeaddim, kIsGroupMode,
///              kIsDeterministic, Traits>
///         -> TileFmhaBwdConvertQGradTraits<spad, dpad, block_per_cu>
///
/// kM0: tile rows along seqlen_q, fixed at 64 (M0_1D in codegen).
template <FmhaBwdConvertDQSpec K>
struct FmhaBwdConvertDQTypes
{
    using QGradDataType    = typename CkTypeMap<K.dtype>::type;
    using QGradAccDataType = float;

    /// kM0 = 64: tile rows along seqlen_q for 1D kernels (OGradDotO, ConvertDQ).
    /// This matches M0_1D in the codegen. The block size (256) is independent;
    /// multiple threads cooperate on each kM0-row tile of width kQKHeaddim.
    static constexpr int kM0 = 64;

    using Traits =
        ck_tile::TileFmhaBwdConvertQGradTraits<K.pad_seqlen_q, K.pad_hdim_q, K.block_per_cu>;

    using PipelineProblem = ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
        QGradAccDataType, // AccDataType (fp32 input)
        QGradDataType,    // QGradDataType (fp16/bf16 output)
        K.block_size,     // BlockSize (256)
        kM0,              // tile rows along seqlen_q
        K.hdim_q,         // kQKHeaddim
        (K.mode == FmhaMode::GROUP),
        K.is_deterministic,
        Traits>;

    using Pipeline = ck_tile::BlockFmhaBwdConvertQGrad<PipelineProblem>;
    using Kernel   = ck_tile::FmhaBwdConvertQGradKernel<Pipeline>;
    using Kargs    = typename Kernel::Kargs;
};

/// Device function that invokes the CK Tile FMHA BWD ConvertDQ kernel.
///
/// Receives generic Args, unpacks tensor pointers and strides via named
/// slot constants, then constructs CK Tile's Kargs via aggregate
/// initialization. MakeKargs() is host-only, so we initialize directly.
///
/// Tensor layout in generic Args (set by host):
///
///   tensors[S::DQ_ACC]: ptr = workspace + GetDqAccDataOffset (host-computed)
///     lengths[0] = seqlen_q
///     lengths[1] = hdim_q
///     lengths[2] = seqlen_k
///     lengths[3] = nhead_q
///     strides[0] = stride_dq             (row stride, index_t)
///
///   tensors[S::DQ]: ptr = dq_ptr (fp16/bf16 output)
///     lengths[0] = seqlen_q
///     lengths[1] = hdim_q
///     strides[0] = stride_dq             (row stride, index_t)
///     strides[1] = nhead_stride_dq       (index_t)
///     strides[2] = batch_stride_dq       (index_t, batch mode only)
///
///   tensors[S::NSPLITS_*]: ptr = workspace + GetDqAccSplitsOffset
///
///   tensors[S::DQ_ACC_BATCH_OFFSET_GROUP]:
///     ptr = workspace + GetDqAccOffsetsOffset (long_index_t* per-batch offsets)
///
///   (group mode only):
///   tensors[S::SEQSTART_Q]: ptr = seqstart_q_ptr  ([batch+1] int32)
///   tensors[S::SEQLEN_Q]:   ptr = seqlen_q_ptr    ([batch] int32, or nullptr)
///   tensors[S::SEQSTART_K]: ptr = seqstart_k_ptr  ([batch+1] int32)
///   tensors[S::SEQLEN_K]:   ptr = seqlen_k_ptr    ([batch] int32, or nullptr)
///
///   No scalar slots are used (ConvertDQ has no scalar parameters).
///
/// Note on const_cast: TensorArg::ptr is const void* for uniform ABI.
/// dq_ptr is an output tensor that the kernel writes to, so we const_cast
/// it to void* here. This is safe because the host allocated it as mutable.
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdConvertDQSpec K>
__device__ void runFmhaBwdConvertDQ(Args args)
{
    using T     = FmhaBwdConvertDQTypes<K>;
    namespace S = fmha_bwd_convert_dq_slots;

    // --- Unpack generic Args using named slot constants ---
    const TensorArg& t_dq_acc = args.tensors[S::DQ_ACC];
    const TensorArg& t_dq     = args.tensors[S::DQ];

    // Dimensions from tensor metadata
    const index_t seqlen_q = t_dq_acc.lengths[0];
    const index_t hdim_q   = t_dq_acc.lengths[1];
    const index_t seqlen_k = t_dq_acc.lengths[2];
    const index_t nhead_q  = t_dq_acc.lengths[3];

    // --- Strides packed into TensorArg::strides[] ---

    const index_t stride_dq       = static_cast<index_t>(t_dq.strides[0]);
    const index_t nhead_stride_dq = static_cast<index_t>(t_dq.strides[1]);

    // nsplits is only read on the deterministic path. Fetching the slot is also
    // gated to keep non-deterministic specs from touching an unused tensor slot.
    const index_t* nsplits_ptr = nullptr;
    if constexpr(K.is_deterministic)
    {
        const TensorArg& t_nsplits = args.tensors[S::nsplitsSlot(K)];
        nsplits_ptr                = reinterpret_cast<const index_t*>(t_nsplits.ptr);
    }

    // --- Construct CK Tile Kargs via aggregate initialization ---
    if constexpr(K.mode == FmhaMode::GROUP)
    {
        // Group mode: variable-length sequences.
        // seqlen_q and seqlen_k are set to -1 (updated per-batch from
        // seqstart/seqlen pointers on the device).
        const TensorArg& t_seqstart_q          = args.tensors[S::SEQSTART_Q];
        const TensorArg& t_seqlen_q            = args.tensors[S::SEQLEN_Q];
        const TensorArg& t_seqstart_k          = args.tensors[S::SEQSTART_K];
        const TensorArg& t_seqlen_k            = args.tensors[S::SEQLEN_K];
        const TensorArg& t_dq_acc_batch_offset = args.tensors[S::DQ_ACC_BATCH_OFFSET_GROUP];

        const long_index_t* dq_acc_batch_offset_ptr =
            reinterpret_cast<const long_index_t*>(t_dq_acc_batch_offset.ptr);

        // Default-init the conditional Deterministic/Empty base with `{}` and assign
        // nsplits_ptr by name under `if constexpr(K.is_deterministic)` below. This
        // mirrors CK Tile's own MakeKargs pattern and stays well-formed regardless of
        // whether the conditional base resolves to FmhaBwdConvertQGradDeterministicKargs
        // (1 field) or FmhaBwdConvertQGradEmptyKargs<0> (0 fields).
        typename T::Kargs kargs{
            // FmhaBwdConvertQGradCommonKargs base
            {t_dq_acc.ptr,                // dq_acc_ptr
             const_cast<void*>(t_dq.ptr), // dq_ptr (output, see const_cast note)
             nhead_q,                     // nhead_q
             -1,                          // seqlen_q (updated per-batch)
             -1,                          // seqlen_k (updated per-batch)
             hdim_q,                      // hdim_q
             stride_dq,                   // stride_dq
             nhead_stride_dq},            // nhead_stride_dq
            {},                           // conditional Deterministic/Empty base
            // FmhaBwdConvertQGradGroupModeKargs extension
            reinterpret_cast<const int32_t*>(t_seqstart_q.ptr),
            reinterpret_cast<const int32_t*>(t_seqstart_k.ptr),
            reinterpret_cast<const int32_t*>(t_seqlen_q.ptr),
            reinterpret_cast<const int32_t*>(t_seqlen_k.ptr),
            nullptr, // cu_seqlen_q_ptr (unused)
            nullptr, // cu_seqlen_k_ptr (unused)
            dq_acc_batch_offset_ptr};
        if constexpr(K.is_deterministic)
        {
            kargs.nsplits_ptr = nsplits_ptr;
        }
        (void)seqlen_q;
        (void)seqlen_k;
        typename T::Kernel{}(kargs);
    }
    else
    {
        // Batch mode: fixed-length sequences
        const index_t batch_stride_dq = static_cast<index_t>(t_dq.strides[2]);

        // See group-mode branch above for rationale: default-init the conditional
        // base with `{}` and assign nsplits_ptr by name under if constexpr.
        typename T::Kargs kargs{
            // FmhaBwdConvertQGradCommonKargs base
            {t_dq_acc.ptr,                // dq_acc_ptr
             const_cast<void*>(t_dq.ptr), // dq_ptr (output, see const_cast note)
             nhead_q,                     // nhead_q
             seqlen_q,                    // seqlen_q
             seqlen_k,                    // seqlen_k
             hdim_q,                      // hdim_q
             stride_dq,                   // stride_dq
             nhead_stride_dq},            // nhead_stride_dq
            {},                           // conditional Deterministic/Empty base
            // FmhaBwdConvertQGradBatchModeKargs extension
            batch_stride_dq // batch_stride_dq
        };
        if constexpr(K.is_deterministic)
        {
            kargs.nsplits_ptr = nsplits_ptr;
        }
        typename T::Kernel{}(kargs);
    }
}

} // namespace rocm_ck
