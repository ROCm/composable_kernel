// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Host-side tensor-load primitives for the quant GEMM bridges that reshuffle
 * their weight / scale tensors before the device copy (aquant, abquant, bquant).
 *
 * It provides the small, repeated building blocks -- loading a raw host pointer
 * into a ck_tile::HostTensor with a given layout, and the unconditional pk_int4
 * permute -- plus the two scale-tensor prep steps that aquant/abquant (AQ) and
 * bquant/abquant (BQ) share verbatim apart from how they spell their quant group
 * sizes.
 *
 * B-matrix prep stays in the per-op source: bquant additionally permutes pk_int4
 * B, abquant does not (its only packed type is pk_fp4_t). Whether that asymmetry
 * is correct is an open question for the kernel owners, so it is deliberately
 * left visible in both files rather than folded into one shared branch here.
 *
 * tensor_quant and rowcolquant perform no reshuffles and do not include this
 * header, so they never pull in the (heavier) shuffle utilities.
 */

#ifndef CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP
#define CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP

#include <algorithm>

#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"

#include "quant_bridge_common.hpp"

namespace quant_bridge {

// Load `rows`x`cols` logical elements from a packed host pointer into a
// HostTensor with leading dim `lead`. RowMajor is a compile-time flag because
// ck_tile::host_tensor_descriptor is overloaded on bool_constant<> layout.
//
// For packed types (pk_int4_t/pk_fp4_t; PackedSize=2) the HostTensor holds only
// rows*cols/PackedSize elements and `src` already contains the packed
// representation, so we copy t.size() elements -- copying rows*cols would overrun
// the source buffer and corrupt the heap (the crash the block-scale bring-up hit
// on every i4/fp4 config before this was fixed).
template <bool RowMajor, typename T>
inline ck_tile::HostTensor<T> load_host_tensor(const T* src, int rows, int cols, int lead)
{
    ck_tile::HostTensor<T> t(
        ck_tile::host_tensor_descriptor(rows, cols, lead, ck_tile::bool_constant<RowMajor>{}));
    std::copy(src, src + t.size(), t.begin());
    return t;
}

// Apply the pk_int4 i4x4 permute in place (mirrors run_gemm_quant_example.inc:
// permute_vectors_i4x4_b, applied unconditionally to pk_int4 operands so the
// device i4->fp8/bf8 conversion sees data in the expected 0x75316420 order).
template <typename T>
inline void permute_i4_inplace(ck_tile::HostTensor<T>& t)
{
    ck_tile::permute_vectors_i4x4_b(t);
}

// Host-side AQ prep, shared by the aquant and abquant bridges
// (run_gemm_quant_example.inc:746-751). APreshuffleQuant kernels read AQ in an
// interleaved layout, so reorder it with shuffle_aq; otherwise AQ goes straight
// to the device. Both bridges load AQ row-major [M, QK_A] with leading dim QK_A.
//
// GroupK is a template parameter because the two bridges spell the A quant group
// size differently (QuantGroupSize::kK vs AQuantGroupSize::kK). Returns the
// hipMemcpy status for BRIDGE_HIP_CHECK. The destination is void* so that QT is
// deduced from the host pointer alone -- passing a DeviceBuffer<QT> for a QT*
// parameter would be a deduction failure, since deduction ignores its conversion
// operator.
//
// Note the row-major load: aquant additionally static_asserts that its AQLayout
// really is row-major, which holds because the ccr layout is excluded from the
// preshufflequant path. abquant's AQ can be column-major on the n=128 EightWaves
// fast path, but only ever with APreshuffleQuant off, so this branch is dead
// there and the plain copy is what runs.
template <typename KernelT, ck_tile::index_t GroupK, typename QT>
inline hipError_t prepare_aq_device(const QT* AQ_host, void* AQ_dev, int64_t M, int64_t QK_A)
{
    const std::size_t aq_bytes = elements_to_bytes<QT>(M * QK_A);
    if constexpr(KernelT::APreshuffleQuant)
    {
        const int block_aq_k = static_cast<int>(KernelT::TileK) / static_cast<int>(GroupK);
        auto aq_h            = load_host_tensor<true>(
            AQ_host, static_cast<int>(M), static_cast<int>(QK_A), static_cast<int>(QK_A));
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        return hipMemcpy(AQ_dev, aq_shuffled.data(), aq_bytes, hipMemcpyHostToDevice);
    }
    else
    {
        return hipMemcpy(AQ_dev, AQ_host, aq_bytes, hipMemcpyHostToDevice);
    }
}

// Host-side BQ prep, shared by the bquant and abquant bridges
// (run_gemm_quant_example.inc:794-825). Three cases:
//   (a) PreshuffleB && TiledMMAPermuteN && GroupN==1: bq_permuteN first, then
//       shuffle_bq if BPreshuffleQuant (else use the permuted BQ).
//   (b) BPreshuffleQuant (no permuteN): shuffle_bq only.
//   (c) neither: plain copy of raw BQ straight to device (no host tensor).
// BQ is ColumnMajor [QK_B, QN_B] (leading dim QK_B).
//
// GroupK/GroupN are template parameters because the two bridges spell the B
// quant group size differently (QuantGroupSize::kK/kN vs BQuantGroupSize::kK and
// BGroupSizeN). Returns the hipMemcpy status for BRIDGE_HIP_CHECK; the
// destination is void* for the same deduction reason as prepare_aq_device.
template <typename KernelT, ck_tile::index_t GroupK, ck_tile::index_t GroupN, typename QT>
inline hipError_t prepare_bq_device(const QT* BQ_host, void* BQ_dev, int64_t QK_B, int64_t QN_B)
{
    constexpr bool use_permute_n =
        KernelT::PreshuffleB && KernelT::TiledMMAPermuteN && (GroupN == 1);
    const std::size_t bq_bytes = elements_to_bytes<QT>(QK_B * QN_B);

    if constexpr(use_permute_n || KernelT::BPreshuffleQuant)
    {
        const int block_bq_k = static_cast<int>(KernelT::TileK) / static_cast<int>(GroupK);
        auto bq_h            = load_host_tensor<false>(
            BQ_host, static_cast<int>(QK_B), static_cast<int>(QN_B), static_cast<int>(QK_B));
        if constexpr(use_permute_n)
        {
            auto bq_permuted = ck_tile::bq_permuteN<typename KernelT::BShuffleConfig>(
                bq_h, static_cast<ck_tile::index_t>(GroupN));
            if constexpr(KernelT::BPreshuffleQuant)
            {
                auto bq_shuffled = ck_tile::shuffle_bq(&bq_permuted, block_bq_k);
                return hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice);
            }
            else
            {
                return hipMemcpy(BQ_dev, bq_permuted.data(), bq_bytes, hipMemcpyHostToDevice);
            }
        }
        else // BPreshuffleQuant only
        {
            auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
            return hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice);
        }
    }
    else
    {
        return hipMemcpy(BQ_dev, BQ_host, bq_bytes, hipMemcpyHostToDevice);
    }
}

} // namespace quant_bridge

#endif // CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP
