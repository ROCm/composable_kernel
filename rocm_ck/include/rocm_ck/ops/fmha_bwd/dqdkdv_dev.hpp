// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD dQ/dK/dV. Maps the validated kernel
// descriptor (FmhaBwdDQDKDVSpec) to the CK Tile template chain.
//
// Unpacks generic rocm_ck::Args via named slot constants, then constructs
// CK Tile's Kargs via field-by-field assignment. MakeKargs()/MakeKargsImpl()
// are host-only, so we initialize directly on the device side.
//
// Uses C++20 struct NTTPs: template <FmhaBwdDQDKDVSpec K>.
//
// Compilation boundary:
//   _spec.hpp -- consteval factory + slot constants (both passes)
//   _api.hpp  -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp (this) -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "dqdkdv_dev.hpp requires device compilation." \
       " Host code should include <rocm_ck/ops/fmha_bwd/dqdkdv_api.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/epilogue.hpp>

namespace rocm_ck {

// =========================================================================
// BiasEnum mapping helper
// =========================================================================

/// Map rocm_ck's FmhaBiasType to CK Tile's BlockAttentionBiasEnum.
consteval ck_tile::BlockAttentionBiasEnum biasTypeToCkEnum(FmhaBiasType bt)
{
    switch(bt)
    {
    case FmhaBiasType::NONE: return ck_tile::BlockAttentionBiasEnum::NO_BIAS;
    case FmhaBiasType::ELEMENTWISE: return ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
    case FmhaBiasType::ALIBI: return ck_tile::BlockAttentionBiasEnum::ALIBI;
    }
    return ck_tile::BlockAttentionBiasEnum::NO_BIAS; // unreachable
}

// =========================================================================
// FmhaBwdDQDKDVTypes -- maps kernel descriptor to CK Tile type chain
// =========================================================================

/// Maps a FmhaBwdDQDKDVSpec descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdDQDKDVSpec<Pipeline, KGradEpi, VGradEpi, QGradEpi>
///     -> BlockFmhaBwdDQDKDVPipeline<PipelineProblem>
///       -> BlockFmhaBwdPipelineProblem<Q,K,V,Gemm,LSE,Acc,D,Bias,RandVal,
///              O,dO,dQ,dK,dV,dBias, Shape, isGroup, isDet, Mask, Dropout,
///              useTrLoad, Traits>
///         -> TileFmhaBwdTraits<padQ, padV, BiasEnum, hasBiasGrad, blockPerCu>
///         -> TileFmhaBwdShape<BlockTile, G0BW, G0WT, G1BW, G1WT,
///              G2BW, G2WT, G3BW, G3WT, G4BW, G4WT, maxSeqLenQ>
template <FmhaBwdDQDKDVSpec K>
struct FmhaBwdDQDKDVTypes
{
    // --- Data types ---
    using QDataType     = typename CkTypeMap<K.dtype>::type;
    using KDataType     = QDataType;
    using VDataType     = QDataType;
    using GemmDataType  = QDataType;
    using ODataType     = QDataType;
    using OGradDataType = QDataType;

    // BiasDataType and BiasGradDataType always match the input dtype.
    // The CK Tile pipeline checks BiasEnum at compile time to decide whether
    // to use the bias pointer; the type itself is always the concrete dtype.
    using BiasDataType     = QDataType;
    using BiasGradDataType = QDataType;

    using LSEDataType           = float;
    using AccDataType           = float;
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;

    using QGradDataType = QDataType;
    using KGradDataType = QDataType;
    using VGradDataType = QDataType;

    // --- BiasEnum ---
    static constexpr auto kBiasEnum = biasTypeToCkEnum(K.bias_type);

    // --- Traits ---
    using Traits = ck_tile::TileFmhaBwdTraits<K.pad_hdim_q,    // kPadHeadDimQ (0, 1, or 8)
                                              K.pad_hdim_v,    // kPadHeadDimV (0, 1, or 8)
                                              kBiasEnum,       // BiasEnum
                                              K.has_bias_grad, // kHasBiasGrad
                                              K.block_per_cu>; // kBlockPerCu

    // Guard: wave64 (gfx9) only. The tile configs currently populated in
    // getTileConfig() assume wavefrontSize=64. On wave32 targets (gfx10/11/12)
    // the block_size arithmetic and warp-level intrinsics would be wrong.
    //
    // Toolchain compatibility for the wavefront-size predefine:
    //   - clang <=22 exposes __AMDGCN_WAVEFRONT_SIZE (and, during the
    //     transition, also the double-underscore __AMDGCN_WAVEFRONT_SIZE__);
    //   - clang >=23 dropped both predefines, so fall back to the gfx9 arch
    //     macro -- gfx9 (CDNA) is wave64 by construction. wave32 targets
    //     (gfx10/11/12) define __GFX10__/__GFX11__/__GFX12__ instead and hit
    //     the #else, rejecting the bridge as intended.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
    static_assert(__AMDGCN_WAVEFRONT_SIZE__ == 64,
                  "FmhaBwdDQDKDV bridge requires wave64 (gfx9). "
                  "Add GFX11/GFX12 tile configs to enable wave32 targets.");
#elif defined(__AMDGCN_WAVEFRONT_SIZE)
    static_assert(__AMDGCN_WAVEFRONT_SIZE == 64,
                  "FmhaBwdDQDKDV bridge requires wave64 (gfx9). "
                  "Add GFX11/GFX12 tile configs to enable wave32 targets.");
#elif defined(__GFX9__)
    // gfx9 is wave64 by construction; no value to assert.
#else
    static_assert(false,
                  "FmhaBwdDQDKDV bridge requires wave64 (gfx9): target is not "
                  "gfx9 and no __AMDGCN_WAVEFRONT_SIZE predefine is available.");
#endif

    // --- Tile shape (from consteval lookup table) ---
    // getTileConfig() returns the architecture-specific tile geometry for
    // the given (hdim_q, hdim_v, dtype). The device bridge reads plain
    // integer fields and converts them to CK Tile sequence<> types.
    static constexpr auto kTile = getTileConfig(K.hdim_q, K.hdim_v, K.dtype, GpuTarget::gfx942);

    using BlockTile = ck_tile::sequence<kTile.bm0,
                                        kTile.bn0,
                                        kTile.bk0,
                                        kTile.bk1,
                                        kTile.bk2,
                                        kTile.bk3,
                                        kTile.bk4,
                                        K.hdim_q,
                                        K.hdim_v>;

    // Gemm0 & Gemm2: compute S = Q @ K^T and dP = dO @ V^T
    using Gemm0BlockWarps = ck_tile::sequence<kTile.rm0, kTile.rn0, kTile.rk0>;
    using Gemm0WarpTile   = ck_tile::sequence<kTile.wm0, kTile.wn0, kTile.wk0>;

    // Gemm1 & Gemm3: compute dV = P^T @ dO and dK = dS^T @ Q
    using Gemm1BlockWarps = ck_tile::sequence<kTile.rm1, kTile.rn1, kTile.rk1>;
    using Gemm1WarpTile   = ck_tile::sequence<kTile.wm1, kTile.wn1, kTile.wk1>;

    // Gemm4: compute dQ = dS @ K
    using Gemm4BlockWarps = ck_tile::sequence<kTile.rm2, kTile.rn2, kTile.rk2>;
    // GEMM4 warp tile = (wm0, wn0, min(wk0, bk4)) per fmha_bwd.py
    static constexpr int kGemm4Wk = (kTile.wk0 < kTile.bk4) ? kTile.wk0 : kTile.bk4;
    using Gemm4WarpTile           = ck_tile::sequence<kTile.wm0, kTile.wn0, kGemm4Wk>;

    // TileFmhaBwdShape: 5 GEMMs with their block_warps and warp_tiles
    //   G0=G2 (S/dP), G1=G3 (dV/dK), G4 (dQ)
    using FmhaShape = ck_tile::TileFmhaBwdShape<BlockTile,
                                                Gemm0BlockWarps,
                                                Gemm0WarpTile, // Gemm0
                                                Gemm1BlockWarps,
                                                Gemm1WarpTile, // Gemm1
                                                Gemm0BlockWarps,
                                                Gemm0WarpTile, // Gemm2 (same as G0)
                                                Gemm1BlockWarps,
                                                Gemm1WarpTile, // Gemm3 (same as G1)
                                                Gemm4BlockWarps,
                                                Gemm4WarpTile,    // Gemm4
                                                kTile.max_seq_q>; // kMaxSeqLenQ

    // --- Mask type ---
    // mask_type != NO_MASK -> GenericAttentionMask<true, true> (full masking)
    // mask_type == NO_MASK -> GenericAttentionMask<false>      (no masking)
    using Mask = std::conditional_t<hasMask(K),
                                    ck_tile::GenericAttentionMask<true, true>,
                                    ck_tile::GenericAttentionMask<false>>;

    // --- Dropout type ---
    // BlockDropoutBwd<IsDropout, IsWG32, IsStoreRandval>
    // For d128 the warp tile wm0=16, so IsWG32=false -> wg16 variant.
    // We never store randval in backward (IsStoreRandval=false).
    using Dropout = ck_tile::BlockDropoutBwd<K.has_dropout,
                                             false,  // IsWG32 = false (wm0=16 for d128 config)
                                             false>; // IsStoreRandval = false

    // --- Pipeline problem ---
    static constexpr bool kIsGroupMode = (K.mode == FmhaMode::GROUP);
    static constexpr bool kUseTrLoad   = false; // non-trload for gfx9

    using PipelineProblem = ck_tile::BlockFmhaBwdPipelineProblem<QDataType,
                                                                 KDataType,
                                                                 VDataType,
                                                                 GemmDataType,
                                                                 LSEDataType,
                                                                 AccDataType,
                                                                 DDataType,
                                                                 BiasDataType,
                                                                 RandValOutputDataType,
                                                                 ODataType,
                                                                 OGradDataType,
                                                                 QGradDataType,
                                                                 KGradDataType,
                                                                 VGradDataType,
                                                                 BiasGradDataType,
                                                                 FmhaShape,
                                                                 kIsGroupMode,
                                                                 K.is_deterministic,
                                                                 Mask,
                                                                 Dropout,
                                                                 kUseTrLoad,
                                                                 Traits>;

    // --- Pipeline (auto-selected by BlockFmhaBwdDQDKDVPipeline) ---
    using Pipeline = ck_tile::BlockFmhaBwdDQDKDVPipeline<PipelineProblem>;

    // --- Epilogues ---
    // KGrad epilogue: AccDataType -> KGradDataType, kPadM=false, kPadN=(padQ>0)
    using KGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     KGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_q > 0) // kPadN
                                                                     >>;

    // VGrad epilogue: AccDataType -> VGradDataType, kPadM=false, kPadN=(padV>0)
    using VGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     VGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_v > 0) // kPadN
                                                                     >>;

    // QGrad epilogue: AccDataType -> QGradDataType, kPadM=false, kPadN=(padQ>0)
    using QGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     QGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_q > 0) // kPadN
                                                                     >>;

    // --- Kernel ---
    using Kernel =
        ck_tile::FmhaBwdDQDKDVKernel<Pipeline, KGradEpilogue, VGradEpilogue, QGradEpilogue>;

    using Kargs = typename Kernel::Kargs;
};

// =========================================================================
// runFmhaBwdDQDKDV -- device function
// =========================================================================

/// Device function that invokes the CK Tile FMHA BWD dQ/dK/dV kernel.
///
/// Receives generic Args, unpacks tensor pointers and strides via named
/// slot constants, then constructs CK Tile's Kargs via field-by-field
/// assignment. MakeKargsImpl() is host-only (uses std::variant for dropout),
/// so we initialize directly on the device side.
///
/// Tensor layout in generic Args (set by host):
///
///   tensors[S::Q]:      ptr, strides=[stride_q, nhead_stride_q, batch_stride_q]
///   tensors[S::K]:      ptr, strides=[stride_k, nhead_stride_k, batch_stride_k]
///   tensors[S::V]:      ptr, strides=[stride_v, nhead_stride_v, batch_stride_v]
///   tensors[S::LSE]:    ptr, strides=[nhead_stride_lsed, batch_stride_lsed]
///   tensors[S::DO]:     ptr, strides=[stride_do, nhead_stride_do, batch_stride_do]
///   tensors[S::D]:      ptr, strides=[nhead_stride_lsed, batch_stride_lsed]
///   tensors[S::DQ_ACC]: ptr, strides=[stride_dq_acc, nhead_stride_dq_acc,
///                                     batch_stride_dq_acc, split_stride_dq_acc]
///   tensors[S::DK]:     ptr, strides=[stride_dk, nhead_stride_dk, batch_stride_dk]
///   tensors[S::DV]:     ptr, strides=[stride_dv, nhead_stride_dv, batch_stride_dv]
///
///   Optional:
///   tensors[S::BIAS]:   ptr, strides=[stride_bias, nhead_stride_bias, batch_stride_bias]
///   tensors[S::DBIAS]:  ptr, strides=[stride_dbias, nhead_stride_dbias, batch_stride_dbias]
///   tensors[S::RANDVAL]: ptr, strides=[stride_randval, nhead_stride_randval,
///                                      batch_stride_randval]
///
///   Each tensor carries its own natural dimensions in lengths[]:
///     Q:  lengths[0]=seqlen_q, lengths[1]=hdim_q
///     K:  lengths[0]=seqlen_k, lengths[1]=hdim_q
///     V:  lengths[0]=seqlen_k, lengths[1]=hdim_v
///
///   Problem-level dims are passed as scalars (not overloaded in lengths):
///     scalars[S::NUM_HEAD_Q].i32     = number of Q heads
///     scalars[S::NHEAD_RATIO_QK].i32 = Q heads / K heads (GQA/MQA)
///
///   scalars[S::RAW_SCALE].f32 = raw_scale (1/sqrt(hdim))
///   scalars[S::SCALE].f32     = raw_scale * log2(e)
///
///   (dropout only):
///   scalars[S::P_UNDROP].f32    = 1/(1-dropout_rate) (CK Tile rp_undrop)
///   scalars[S::RP_UNDROP].f32   = 1 - dropout_rate   (keep_prob)
///   scalars[S::DROP_SEED].u64   = dropout RNG seed
///   scalars[S::DROP_OFFSET].u64 = dropout RNG offset
///
/// CK Tile stores row strides as index_t (int32). Large strides that
/// exceed INT32_MAX will be silently truncated -- a known CK Tile limitation.
/// The nhead_stride_dq_acc is long_index_t (int64).
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdDQDKDVSpec K>
__device__ void runFmhaBwdDQDKDV(Args args)
{
    using T     = FmhaBwdDQDKDVTypes<K>;
    namespace S = fmha_bwd_dqdkdv_slots;

    // --- Unpack generic Args using named slot constants ---
    const TensorArg& t_q      = args.tensors[S::Q];
    const TensorArg& t_k      = args.tensors[S::K];
    const TensorArg& t_v      = args.tensors[S::V];
    const TensorArg& t_lse    = args.tensors[S::LSE];
    const TensorArg& t_do     = args.tensors[S::DO];
    const TensorArg& t_d      = args.tensors[S::D];
    const TensorArg& t_dq_acc = args.tensors[S::DQ_ACC];
    const TensorArg& t_dk     = args.tensors[S::DK];
    const TensorArg& t_dv     = args.tensors[S::DV];

    // --- Dimensions from tensor metadata and scalars ---
    // Each tensor carries its own natural dimensions in lengths[].
    // Problem-level dims (num_head_q, nhead_ratio_qk) are scalars.
    const index_t seqlen_q       = t_q.lengths[0]; // Q: [seqlen_q, hdim_q]
    const index_t hdim_q         = t_q.lengths[1];
    const index_t seqlen_k       = t_k.lengths[0]; // K: [seqlen_k, hdim_q]
    const index_t hdim_v         = t_v.lengths[1]; // V: [seqlen_k, hdim_v]
    const index_t num_head_q     = args.scalars[S::NUM_HEAD_Q].i32;
    const index_t nhead_ratio_qk = args.scalars[S::NHEAD_RATIO_QK].i32;

    // --- Scalars ---
    const float raw_scale = args.scalars[S::RAW_SCALE].f32;
    const float scale     = args.scalars[S::SCALE].f32;

    // --- Strides ---
    // Q: strides[0]=stride_q, [1]=nhead_stride_q, [2]=batch_stride_q
    const index_t stride_q       = static_cast<index_t>(t_q.strides[0]);
    const index_t nhead_stride_q = static_cast<index_t>(t_q.strides[1]);

    // K: strides[0]=stride_k, [1]=nhead_stride_k, [2]=batch_stride_k
    const index_t stride_k       = static_cast<index_t>(t_k.strides[0]);
    const index_t nhead_stride_k = static_cast<index_t>(t_k.strides[1]);

    // V: strides[0]=stride_v, [1]=nhead_stride_v, [2]=batch_stride_v
    const index_t stride_v       = static_cast<index_t>(t_v.strides[0]);
    const index_t nhead_stride_v = static_cast<index_t>(t_v.strides[1]);

    // LSE: strides[0]=nhead_stride_lsed, [1]=batch_stride_lsed
    const index_t nhead_stride_lsed = static_cast<index_t>(t_lse.strides[0]);

    // DO: strides[0]=stride_do, [1]=nhead_stride_do, [2]=batch_stride_do
    const index_t stride_do       = static_cast<index_t>(t_do.strides[0]);
    const index_t nhead_stride_do = static_cast<index_t>(t_do.strides[1]);

    // DQ_ACC carries no stride fields in FmhaBwdCommonKargs: the DqDkDv kernel
    // derives the dq_acc layout internally (compact split-K layout computed from
    // hdim_q/seqlen/nsplits). Only t_dq_acc.ptr is consumed below; its strides
    // in the generic Args are reserved for the ConvertDQ kernel.

    // DK: strides[0]=stride_dk, [1]=nhead_stride_dk, [2]=batch_stride_dk
    const index_t stride_dk       = static_cast<index_t>(t_dk.strides[0]);
    const index_t nhead_stride_dk = static_cast<index_t>(t_dk.strides[1]);

    // DV: strides[0]=stride_dv, [1]=nhead_stride_dv, [2]=batch_stride_dv
    const index_t stride_dv       = static_cast<index_t>(t_dv.strides[0]);
    const index_t nhead_stride_dv = static_cast<index_t>(t_dv.strides[1]);

    // --- Construct CK Tile Kargs ---
    //
    // The Kargs struct uses multiple inheritance with conditional base classes.
    // Aggregate initialization must match the exact inheritance order:
    //   1. FmhaBwdCommonKargs                (30 fields, common to both modes)
    //   2..6. Bias / BiasGrad / Mask / Dropout / Deterministic kargs
    //         (each is either FmhaBwdEmptyKargs<N> or its full counterpart,
    //          chosen at compile time by the pipeline problem traits)
    //   7. Batch- or Group-mode tail members (fixed-length vs variable-length
    //      sequence metadata)
    //
    // Strategy: aggregate-init the COMMON base positionally (its layout is
    // stable inside CK Tile), pass `{}` for the 5 optional bases (zero-init
    // matches FmhaBwdEmptyKargs<N> and is a sane default for the populated
    // variants -- we fill them by name below), and leave the mode-specific
    // tail value-initialized so we can assign each tail field by name. This
    // mirrors CK Tile's own MakeKargsImpl while removing the positional
    // fragility flagged by the W7 finding.

    typename T::Kargs kargs{
        // FmhaBwdCommonKargs (30 positional fields)
        {t_q.ptr,                         // q_ptr
         t_k.ptr,                         // k_ptr
         t_v.ptr,                         // v_ptr
         t_lse.ptr,                       // lse_ptr
         t_do.ptr,                        // do_ptr
         t_d.ptr,                         // d_ptr (input: const void*)
         const_cast<void*>(t_dq_acc.ptr), // dq_acc_ptr (output)
         const_cast<void*>(t_dk.ptr),     // dk_ptr (output)
         const_cast<void*>(t_dv.ptr),     // dv_ptr (output)
         // Group mode passes -1 for both seqlens; CK Tile reads per-batch
         // lengths from seqstart/seqlen pointers below. Batch mode passes
         // the fixed sequence dimensions.
         (K.mode == FmhaMode::GROUP) ? index_t{-1} : seqlen_q, // seqlen_q
         (K.mode == FmhaMode::GROUP) ? index_t{-1} : seqlen_k, // seqlen_k
         hdim_q,                                               // hdim_q
         hdim_v,                                               // hdim_v
         num_head_q,                                           // num_head_q
         nhead_ratio_qk,                                       // nhead_ratio_qk
         raw_scale,                                            // raw_scale
         scale,                                                // scale
         stride_q,                                             // stride_q
         stride_k,                                             // stride_k
         stride_v,                                             // stride_v
         stride_do,                                            // stride_do
         stride_dk,                                            // stride_dk
         stride_dv,                                            // stride_dv
         nhead_stride_q,                                       // nhead_stride_q
         nhead_stride_k,                                       // nhead_stride_k
         nhead_stride_v,                                       // nhead_stride_v
         nhead_stride_do,                                      // nhead_stride_do
         nhead_stride_lsed,                                    // nhead_stride_lsed
         nhead_stride_dk,                                      // nhead_stride_dk
         nhead_stride_dv},                                     // nhead_stride_dv
        {}, // bias placeholder    (EmptyKargs<0> or BiasKargs)
        {}, // dbias placeholder   (EmptyKargs<1> or BiasGradKargs)
        {}, // mask placeholder    (EmptyKargs<2> or MaskKargs)
        {}, // dropout placeholder (EmptyKargs<3> or DropoutKargs)
        {}, // determ placeholder  (EmptyKargs<4> or DetermKargs)
        {}, // qrqtrdor (QrQtrDorKargs when kUseQrQtrDorPipeline; unused here)
        // Mode-specific tail fields are value-initialized (zero) here and
        // assigned by name in the per-mode block below.
    };

    // --- Mode-specific tail (named-field assignment, no positional drift) ---

    if constexpr(K.mode == FmhaMode::GROUP)
    {
        // Variable-length sequences: seqstart pointers give per-batch offsets
        // into the packed Q/K/V buffers; seqlen pointers carry actual lengths.
        // cu_seqlen_*_ptr are deprecated CK Tile fields (seqstart/seqlen are
        // authoritative); leave them null.
        kargs.seqstart_q_ptr  = reinterpret_cast<const int32_t*>(args.tensors[S::SEQSTART_Q].ptr);
        kargs.seqstart_k_ptr  = reinterpret_cast<const int32_t*>(args.tensors[S::SEQSTART_K].ptr);
        kargs.seqlen_q_ptr    = reinterpret_cast<const int32_t*>(args.tensors[S::SEQLEN_Q].ptr);
        kargs.seqlen_k_ptr    = reinterpret_cast<const int32_t*>(args.tensors[S::SEQLEN_K].ptr);
        kargs.cu_seqlen_q_ptr = nullptr;
        kargs.cu_seqlen_k_ptr = nullptr;
        // dq_acc_batch_offset_ptr: per-batch workspace offset table, populated
        // via workspace. nullptr when !is_deterministic (host won't populate
        // the slot and DqDkDv kernel doesn't read it).
        if constexpr(K.is_deterministic)
            kargs.dq_acc_batch_offset_ptr =
                reinterpret_cast<const long_index_t*>(args.tensors[S::DQ_ACC_BATCH_OFFSET].ptr);
        else
            kargs.dq_acc_batch_offset_ptr = nullptr;
    }
    else
    {
        // Fixed-length sequences: per-batch strides for every tensor.
        // dq_acc layout derived from workspace via nsplits_ptr
        kargs.batch_stride_q    = static_cast<index_t>(t_q.strides[2]);
        kargs.batch_stride_k    = static_cast<index_t>(t_k.strides[2]);
        kargs.batch_stride_v    = static_cast<index_t>(t_v.strides[2]);
        kargs.batch_stride_do   = static_cast<index_t>(t_do.strides[2]);
        kargs.batch_stride_lsed = static_cast<index_t>(t_lse.strides[1]);
        kargs.batch_stride_dk   = static_cast<index_t>(t_dk.strides[2]);
        kargs.batch_stride_dv   = static_cast<index_t>(t_dv.strides[2]);
    }

    // --- Optional feature fields (single set of if-constexpr branches) ---
    //
    // Bias / dbias batch_stride_* and dropout batch_stride_randval fields
    // exist only on the BATCH-mode arms of the optional kargs. Guarding by
    // K.mode prevents referencing absent members in GROUP mode. CK Tile's
    // GROUP-mode optional kargs use the common forms instead: bias/dbias are
    // addressed through stride + nhead_stride without batch_stride, dropout
    // uses randval stride + nhead_stride without batch_stride_randval, and
    // batch offsets come from seqstart_q_ptr / seqstart_k_ptr.

    if constexpr(K.bias_type == FmhaBiasType::ELEMENTWISE)
    {
        const TensorArg& t_bias = args.tensors[S::BIAS];
        kargs.bias_ptr          = t_bias.ptr;
        kargs.stride_bias       = static_cast<index_t>(t_bias.strides[0]);
        kargs.nhead_stride_bias = static_cast<index_t>(t_bias.strides[1]);
        if constexpr(K.mode == FmhaMode::BATCH)
        {
            kargs.batch_stride_bias = static_cast<index_t>(t_bias.strides[2]);
        }
    }
    else if constexpr(K.bias_type == FmhaBiasType::ALIBI)
    {
        const TensorArg& t_bias  = args.tensors[S::BIAS];
        kargs.alibi_slope_ptr    = t_bias.ptr;
        kargs.alibi_slope_stride = static_cast<index_t>(t_bias.strides[0]);
    }

    if constexpr(K.has_bias_grad)
    {
        const TensorArg& t_dbias = args.tensors[S::DBIAS];
        kargs.dbias_ptr          = const_cast<void*>(t_dbias.ptr);
        kargs.stride_dbias       = static_cast<index_t>(t_dbias.strides[0]);
        kargs.nhead_stride_dbias = static_cast<index_t>(t_dbias.strides[1]);
        if constexpr(K.mode == FmhaMode::BATCH)
        {
            kargs.batch_stride_dbias = static_cast<index_t>(t_dbias.strides[2]);
        }
    }

    if constexpr(hasMask(K))
    {
        kargs.window_size_left  = args.scalars[S::WINDOW_SIZE_LEFT].i32;
        kargs.window_size_right = args.scalars[S::WINDOW_SIZE_RIGHT].i32;
        kargs.mask_type =
            static_cast<ck_tile::GenericAttentionMaskEnum>(args.scalars[S::MASK_TYPE].i32);
    }

    if constexpr(K.has_dropout)
    {
        const TensorArg& t_randval = args.tensors[S::RANDVAL];
        // rocm_ck's historical slot names are inverted relative to CK Tile's
        // kargs naming: P_UNDROP stores the reciprocal keep probability used
        // as CK Tile's rp_undrop, while RP_UNDROP stores the keep probability
        // used for the uint8 dropout threshold. (Prior to PR #7534 these two
        // slots were assigned reversed -- kargs.rp_undrop received the keep
        // probability instead of its reciprocal -- which produced incorrect
        // gradient scaling for any non-trivial dropout rate. The bug was
        // masked because every landed host caller wrote 1.0 to both slots,
        // making the swap unobservable in tests.)
        const float rp_undrop = args.scalars[S::P_UNDROP].f32;
        const float p_undrop  = args.scalars[S::RP_UNDROP].f32;
        kargs.rp_undrop       = rp_undrop;
        kargs.scale_rp_undrop = rp_undrop * raw_scale;
        // Clamp before the uint8_t cast: out-of-range float-to-int
        // conversion is UB. Host setup may pass p_undrop > 1.0 for
        // dropout-disabled "no-drop" launches.
        const float p_undrop_byte =
            __builtin_fmaxf(0.0f, __builtin_fminf(255.0f, __builtin_floorf(p_undrop * 255.0f)));
        kargs.p_undrop_in_uint8_t = static_cast<uint8_t>(p_undrop_byte);

        kargs.drop_seed.val                 = args.scalars[S::DROP_SEED].u64;
        kargs.drop_offset.val               = args.scalars[S::DROP_OFFSET].u64;
        kargs.is_drop_seed_offset_from_host = true;

        kargs.rand_val_ptr         = const_cast<void*>(t_randval.ptr);
        kargs.stride_randval       = static_cast<index_t>(t_randval.strides[0]);
        kargs.nhead_stride_randval = static_cast<index_t>(t_randval.strides[1]);
        if constexpr(K.mode == FmhaMode::BATCH)
        {
            kargs.batch_stride_randval = static_cast<index_t>(t_randval.strides[2]);
        }
    }

    if constexpr(K.is_deterministic)
    {
        // FmhaBwdDeterministicKargs = {batch, nsplits_ptr}.
        // - batch: only used by the persistent batch-mode kernel (computes
        //   total_heads = batch * nhead_q). In group mode the value is
        //   ignored. Sourced from S::BATCH_SIZE scalar.
        // - nsplits_ptr: workspace + GetDqAccSplitsOffset, pre-filled by
        //   PrepareWorkspaceHost.
        kargs.batch = (K.mode == FmhaMode::BATCH) ? args.scalars[S::BATCH_SIZE].i32 : index_t{0};
        kargs.nsplits_ptr = reinterpret_cast<const index_t*>(args.tensors[S::NSPLITS].ptr);
    }

    typename T::Kernel{}(kargs);
}

} // namespace rocm_ck
