// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
 *
 * C API for the plain (non-grouped) B-only quantized block-scale GEMM operator
 * from example/ck_tile/38_block_scale_gemm. One .so per kernel variant; the
 * kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_bquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType, QuantGroupSize.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * Shared infrastructure lives in quant_bridge_common.hpp; host-load primitives
 * in quant_bridge_shuffle.hpp. Memory model: host-pointer.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "quant_bridge_common.hpp"
#include "quant_bridge_shuffle.hpp"

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run non-grouped BQuant GEMM:
 *   C[M,N] = A[M,K] @ dequant(B[K,N], BQ[ceil(K/gK), ceil(N/gN)])
 * A, B, BQ, C are host pointers; device memory is managed internally. QK_B /
 * QN_B are the number of K / N groups. time_ms is optional. Returns 0 on success.
 */
int dispatcher_run_bquant_gemm(const void* A,
                               const void* B,
                               const void* BQ,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t K,
                               int64_t stride_A,
                               int64_t stride_B,
                               int64_t stride_BQ,
                               int64_t stride_C,
                               int64_t QK_B,
                               int64_t QN_B,
                               int k_batch,
                               float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_bquant_gemm";

    if(!check_entry_args(kFn,
                         g_initialized,
                         {A, B, BQ, C},
                         {M, N, K, QK_B, QN_B},
                         /*allow_gfx90a=*/true))
        return -1;

    // Validate QK_B/QN_B against the compile-time quant group sizes baked into this .so.
    if(!check_quant_group_count(kFn, "QK_B", QK_B, "K", K, QuantGroupSize::kK) ||
       !check_quant_group_count(kFn, "QN_B", QN_B, "N", N, QuantGroupSize::kN))
        return -1;

    // Only packed layouts are supported. BQ is ColumnMajor [QK_B, QN_B] (leading
    // dim QK_B), matching Old-TE's rcr path and the WPQuantB pipeline.
    if(stride_A != K || stride_B != K || stride_BQ != QK_B || stride_C != N)
    {
        std::cerr << kFn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_BQ=" << QK_B << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_BQ=" << stride_BQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);

    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> BQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, BQ_dev.allocate(elements_to_bytes<QDataType>(QK_B * QN_B)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));

    // Host-side B prep (run_gemm_quant_example.inc:770-789): only touch host
    // memory when a reshuffle or pk_int4 permute is actually applied. PreshuffleB
    // kernels pre-shuffle B into the interleaved layout the WPQuantB pipeline
    // reads (shuffle_b_permuteN when TiledMMAPermuteN && kN==1, else shuffle_b);
    // pk_int4 B is permute_i4_inplace'd. The common (no-preshuffle, unpacked)
    // path copies raw B straight to device with no intermediate host tensors.
    if constexpr(SelectedKernel::PreshuffleB || std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {
        auto b_k_n = load_host_tensor<false>(
            B_host, static_cast<int>(K), static_cast<int>(N), static_cast<int>(K));
        if constexpr(SelectedKernel::PreshuffleB)
        {
            constexpr bool use_permute_n =
                SelectedKernel::TiledMMAPermuteN && (QuantGroupSize::kN == 1);
            auto b_shuffled = [&]() {
                if constexpr(use_permute_n)
                    return ck_tile::shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>(
                        b_k_n);
                else
                    return ck_tile::shuffle_b<typename SelectedKernel::BShuffleConfig>(b_k_n);
            }();
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
                permute_i4_inplace(b_shuffled);
            BRIDGE_HIP_CHECK(kFn,
                             hipMemcpy(B_dev,
                                       b_shuffled.data(),
                                       elements_to_bytes<BDataType>(K * N),
                                       hipMemcpyHostToDevice));
        }
        else // pk_int4 B, no preshuffle
        {
            permute_i4_inplace(b_k_n);
            BRIDGE_HIP_CHECK(kFn,
                             hipMemcpy(B_dev,
                                       b_k_n.data(),
                                       elements_to_bytes<BDataType>(K * N),
                                       hipMemcpyHostToDevice));
        }
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }

    // Host-side BQ prep (bq_permuteN / shuffle_bq / plain copy); shared verbatim
    // with abquant, see prepare_bq_device().
    BRIDGE_HIP_CHECK(kFn,
                     (prepare_bq_device<SelectedKernel, QuantGroupSize::kK, QuantGroupSize::kN>(
                         BQ_host, BQ_dev, QK_B, QN_B)));
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // BQuant-only: aq_ptr = nullptr, QK_A = 0, stride_AQ = 0.
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = nullptr;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = 0;
    args.QK_B      = static_cast<ck_tile::index_t>(QK_B);
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 0;
    args.stride_BQ = static_cast<ck_tile::index_t>(stride_BQ);

    return launch_and_copyback<SelectedKernel, CDataType>(
        kFn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // extern "C"
