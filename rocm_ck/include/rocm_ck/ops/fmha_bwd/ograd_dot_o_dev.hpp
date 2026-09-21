// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD OGradDotO. Maps the validated kernel
// descriptor (FmhaBwdOGradDotOSpec) to the CK Tile template chain.
//
// Unpacks generic rocm_ck::Args via named slot constants, then constructs
// CK Tile's Kargs via aggregate initialization. No __builtin_bit_cast,
// no ABI matching required.
//
// Uses C++20 struct NTTPs: template <FmhaBwdOGradDotOSpec K>.
//
// Compilation boundary:
//   _spec.hpp -- consteval factory + slot constants (both passes)
//   _api.hpp  -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp (this) -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "ograd_dot_o_dev.hpp requires device compilation." \
       " Host code should include <rocm_ck/ops/fmha_bwd/ograd_dot_o_api.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>

namespace rocm_ck {

/// Maps a FmhaBwdOGradDotOSpec descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdOGradDotOSpec<Pipeline>
///     -> BlockFmhaBwdOGradDotO<PipelineProblem>
///       -> BlockFmhaBwdOGradDotOPipelineProblem<OType, dOType, DType, LSEType,
///              bm0, hdim_v, is_group, Traits>
///         -> TileFmhaBwdOGradDotOTraits<spad, dvpad, block_per_cu>
///
/// LSEDataType is fixed to float to satisfy the sink-gradient atomicAdd
/// static_assert in block_fmha_bwd_dot_do_o.hpp
template <FmhaBwdOGradDotOSpec K>
struct FmhaBwdOGradDotOTypes
{
    using ODataType     = typename CkTypeMap<K.dtype>::type;
    using OGradDataType = ODataType;
    using DDataType     = float;
    using LSEDataType   = float;

    using Traits =
        ck_tile::TileFmhaBwdOGradDotOTraits<K.pad_seqlen_q, K.pad_hdim_v, K.block_per_cu>;

    using PipelineProblem =
        ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<ODataType,
                                                      OGradDataType,
                                                      DDataType,
                                                      LSEDataType,
                                                      K.block_size,
                                                      K.hdim_v,
                                                      (K.mode == FmhaMode::GROUP),
                                                      Traits>;

    using Pipeline = ck_tile::BlockFmhaBwdOGradDotO<PipelineProblem>;
    using Kernel   = ck_tile::FmhaBwdOGradDotOKernel<Pipeline>;
    using Kargs    = typename Kernel::Kargs;
};

/// Device function that invokes the CK Tile FMHA BWD OGradDotO kernel.
///
/// Receives generic Args, unpacks tensor pointers and strides via named
/// slot constants, then constructs CK Tile's Kargs via aggregate
/// initialization. MakeKargs() is host-only, so we initialize directly.
///
/// Sink-token gradient (CK Tile #5504) is disabled in rocm_ck: lse_ptr,
/// sink_ptr and d_sink_ptr are all passed as nullptr and nhead as 0. This is
/// safe: the kernel only *loads* LSE when atomic_sink_grad_ptr != nullptr (i.e.
/// when d_sink_ptr != nullptr), and the sink-score read is guarded by
/// sink_ptr != nullptr -- so these null pointers are address-computed but never
/// dereferenced. nhead only scales the (never-taken) sink index.
///
/// Tensor layout in generic Args (set by host):
///   tensors[S::O]:  ptr=o_ptr,  strides=[stride_o, nhead_stride_o, batch_stride_o]
///   tensors[S::DO]: ptr=do_ptr, strides=[stride_do, nhead_stride_do, batch_stride_do]
///   tensors[S::D]:  ptr=d_ptr,  strides=[nhead_stride_d, batch_stride_d]
///   (group mode only):
///   tensors[S::SEQSTART_Q]: ptr=seqstart_q_ptr
///   tensors[S::SEQLEN_Q]:   ptr=seqlen_q_ptr (nullptr if unused)
///
///   lengths[0] = seqlen_q, lengths[1] = hdim_v (in O/DO tensors)
///
///   scalars[S::P_UNDROP].f32 = p_undrop
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdOGradDotOSpec K>
__device__ void runFmhaBwdOGradDotO(Args args)
{
    using T     = FmhaBwdOGradDotOTypes<K>;
    namespace S = fmha_bwd_ograd_dot_o_slots;

    // --- Unpack generic Args using named slot constants ---
    const TensorArg& t_o  = args.tensors[S::O];
    const TensorArg& t_do = args.tensors[S::DO];
    const TensorArg& t_d  = args.tensors[S::D];

    const float p_undrop = args.scalars[S::P_UNDROP].f32;

    // Dimensions from tensor metadata
    const index_t seqlen_q = t_o.lengths[0];
    const index_t hdim_v   = t_o.lengths[1];

    // Strides packed into TensorArg::strides[]
    // O/DO: strides[0]=stride, strides[1]=nhead_stride, strides[2]=batch_stride
    // D:    strides[0]=nhead_stride, strides[1]=batch_stride
    // CK Tile stores row strides as index_t (int32). Large strides that
    // exceed INT32_MAX will be silently truncated -- a known CK Tile limitation.
    const index_t stride_o        = static_cast<index_t>(t_o.strides[0]);
    const index_t stride_do       = static_cast<index_t>(t_do.strides[0]);
    const index_t nhead_stride_o  = static_cast<index_t>(t_o.strides[1]);
    const index_t nhead_stride_do = static_cast<index_t>(t_do.strides[1]);
    const index_t nhead_stride_d  = static_cast<index_t>(t_d.strides[0]);

    // --- Construct CK Tile Kargs via aggregate initialization ---
    if constexpr(K.mode == FmhaMode::GROUP)
    {
        // Group mode: variable-length sequences
        const TensorArg& t_seqstart_q = args.tensors[S::SEQSTART_Q];
        const TensorArg& t_seqlen_q   = args.tensors[S::SEQLEN_Q];

        const typename T::Kargs kargs{
            // FmhaBwdOGradDotOCommonKargs base
            {t_o.ptr,                    // o_ptr
             t_do.ptr,                   // do_ptr
             const_cast<void*>(t_d.ptr), // d_ptr (output: TensorArg::ptr is const void*)
             // lse_ptr/sink_ptr/d_sink_ptr: all nullptr (sink-grad disabled in rocm_ck).
             // Safe because the LSE load is gated on (atomic_sink_grad_ptr != nullptr) and
             // the sink read on (sink_ptr != nullptr); with d_sink_ptr null neither fires.
             // CK Tile still computes address arithmetic on lse_ptr unconditionally; a
             // follow-up upstream issue should skip that math when sink-grad is off so this
             // nullptr goes away entirely.
             nullptr,         // lse_ptr
             nullptr,         // sink_ptr
             nullptr,         // d_sink_ptr
             p_undrop,        // p_undrop
             -1,              // seqlen_q (updated per-batch)
             hdim_v,          // hdim_v
             0,               // nhead (only indexes sink_ptr/d_sink_ptr -- 0 when disabled)
             stride_do,       // stride_do
             stride_o,        // stride_o
             nhead_stride_do, // nhead_stride_do
             nhead_stride_o,  // nhead_stride_o
             nhead_stride_d}, // nhead_stride_lsed (D and LSE share layout)
            // FmhaBwdOGradDotOGroupModeKargs extension
            reinterpret_cast<const int32_t*>( // seqstart_q_ptr
                t_seqstart_q.ptr),
            reinterpret_cast<const int32_t*>( // seqlen_q_ptr
                t_seqlen_q.ptr),
            nullptr // cu_seqlen_q_ptr (unused)
        };
        typename T::Kernel{}(kargs);
    }
    else
    {
        // Batch mode: fixed-length sequences
        const index_t batch_stride_o  = static_cast<index_t>(t_o.strides[2]);
        const index_t batch_stride_do = static_cast<index_t>(t_do.strides[2]);
        const index_t batch_stride_d  = static_cast<index_t>(t_d.strides[1]);

        const typename T::Kargs kargs{
            // FmhaBwdOGradDotOCommonKargs base
            {t_o.ptr,                    // o_ptr
             t_do.ptr,                   // do_ptr
             const_cast<void*>(t_d.ptr), // d_ptr (output: TensorArg::ptr is const void*)
             // lse_ptr/sink_ptr/d_sink_ptr: all nullptr (sink-grad disabled in rocm_ck).
             // Safe because the LSE load is gated on (atomic_sink_grad_ptr != nullptr) and
             // the sink read on (sink_ptr != nullptr); with d_sink_ptr null neither fires.
             // CK Tile still computes address arithmetic on lse_ptr unconditionally; a
             // follow-up upstream issue should skip that math when sink-grad is off so this
             // nullptr goes away entirely.
             nullptr,         // lse_ptr
             nullptr,         // sink_ptr
             nullptr,         // d_sink_ptr
             p_undrop,        // p_undrop
             seqlen_q,        // seqlen_q
             hdim_v,          // hdim_v
             0,               // nhead (only indexes sink_ptr/d_sink_ptr -- 0 when disabled)
             stride_do,       // stride_do
             stride_o,        // stride_o
             nhead_stride_do, // nhead_stride_do
             nhead_stride_o,  // nhead_stride_o
             nhead_stride_d}, // nhead_stride_lsed (D and LSE share layout)
            // FmhaBwdOGradDotOBatchModeKargs extension
            batch_stride_do, // batch_stride_do
            batch_stride_o,  // batch_stride_o
            batch_stride_d   // batch_stride_lsed (was batch_stride_d; same value)
        };
        typename T::Kernel{}(kargs);
    }
}

} // namespace rocm_ck
