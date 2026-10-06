// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm ABQuant (A+B block-scale) ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_abquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType, AQuantGroupSize, BQuantGroupSize.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * ABQuant quantizes BOTH A and B: aq_ptr AND bq_ptr are non-null. AQ is stored
 * RowMajor [M, QK_A] (QK_A = ceil(K / AGroupSizeK)); BQ is stored ColumnMajor
 * [QK_B, QN_B] (QK_B = ceil(K / BGroupSizeK), QN_B = ceil(N / BGroupSizeN);
 * BQLayout==ColumnMajor is enforced by a static_assert in gemm_quant_kernel.hpp).
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
 * Run ABQuant GEMM:
 *   C[M,N] = dequant(A[M,K], AQ[M,QK_A]) @ dequant(B[K,N], BQ[QK_B,QN_B])
 * A, B, AQ, BQ, C are host pointers; device memory is managed internally. QK_A,
 * QK_B, QN_B are the A K-group / B K-group / B N-group counts. time_ms is
 * optional. Returns 0 on success, negative on error.
 */
int dispatcher_run_abquant_gemm(const void* A,
                                const void* B,
                                const void* AQ,
                                const void* BQ,
                                void* C,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                int64_t stride_A,
                                int64_t stride_B,
                                int64_t stride_AQ,
                                int64_t stride_BQ,
                                int64_t stride_C,
                                int64_t QK_A,
                                int64_t QK_B,
                                int64_t QN_B,
                                int k_batch,
                                float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_abquant_gemm";

    // check_arch=false: unlike the other bridges, the arch check must run *after*
    // the fp4-preshuffle reject below so that case still returns -3, not -1.
    if(!check_entry_args(kFn,
                         g_initialized,
                         {A, B, AQ, BQ, C},
                         {M, N, K, QK_A, QK_B, QN_B},
                         /*allow_gfx90a=*/false,
                         /*check_arch=*/false))
        return -1;

    // ABQuant never needs permute_i4_inplace on B, unlike gemm_bquant which
    // applies it for pk_int4_t B. ABQUANT_VARIANTS admits only fp8/bf8/fp4 --
    // no pk_int4_t. For pk_fp4_t (the only packed 4-bit type abquant uses),
    // the on-device layout is flat fp4x2 pairs, not the i4x4 interleaved tiles
    // that require permute_i4_inplace. No host-side permute is needed.
    static_assert(!std::is_same_v<BDataType, ck_tile::pk_int4_t>,
                  "ABQuant does not support pk_int4_t B -- update B-prep if adding i4 support");

    // Graceful reject: PreshuffleB is not supported for fp4 (BDataType==pk_fp4_t),
    // exactly as Old-TE THROWS in run_gemm_quant_example.inc:994-1001. The fp4
    // preshuffle host path would otherwise allocate/copy a mis-sized B buffer and
    // heap-corrupt. Compile-time branch: can only fire in an fp4 PreshuffleB .so.
    if constexpr(SelectedKernel::PreshuffleB && std::is_same_v<BDataType, ck_tile::pk_fp4_t>)
    {
        std::cerr << kFn
                  << ": Preshuffling weight matrix is not supported for bf16_fp4_gemm "
                     "(matches Old-TE reject)\n";
        return -3;
    }

    if(!validate_supported_arch(kFn))
        return -1;

    // Validate QK_A/QK_B/QN_B against the compile-time quant group sizes.
    if(!check_quant_group_count(kFn, "QK_A", QK_A, "K", K, AQuantGroupSize::kK) ||
       !check_quant_group_count(kFn, "QK_B", QK_B, "K", K, BQuantGroupSize::kK) ||
       !check_quant_group_count(kFn, "QN_B", QN_B, "N", N, BQuantGroupSize::kN))
        return -1;

    // Only packed layouts are supported. A leading dim depends on ALayout: the
    // ccr/crr families use ColumnMajor A [M, K] -> M; rcr/rrr use RowMajor -> K.
    // AQ leading dim depends on AQLayout: the n=128 EightWaves fast path uses
    // ColumnMajor [M, QK_A] -> M; otherwise RowMajor -> QK_A. BQ is ColumnMajor
    // [QK_B, QN_B] -> leading dim QK_B. ALayout is the generated-header typedef
    // (ck_tile::tensor_layout::gemm::{Row,Column}Major), in scope via the bridge
    // namespace -- no new codegen field needed, mirrors AQIsColumnMajor intent.
    constexpr bool kAIsColumnMajor =
        std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>;
    constexpr bool kBIsColumnMajor =
        std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>;
    const int64_t expected_stride_A  = kAIsColumnMajor ? M : K;
    const int64_t expected_stride_B  = kBIsColumnMajor ? K : N; // ColMajor->K, RowMajor->N
    const int64_t expected_stride_AQ = SelectedKernel::AQIsColumnMajor ? M : QK_A;
    if(stride_A != expected_stride_A || stride_B != expected_stride_B ||
       stride_AQ != expected_stride_AQ || stride_BQ != QK_B || stride_C != N)
    {
        std::cerr << kFn << ": non-packed strides are not supported. Expected stride_A="
                  << expected_stride_A << " stride_B=" << expected_stride_B
                  << " stride_AQ=" << expected_stride_AQ << " stride_BQ=" << QK_B
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_AQ=" << stride_AQ << " stride_BQ=" << stride_BQ
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);

    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> AQ_dev;
    DeviceBuffer<QDataType> BQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, AQ_dev.allocate(elements_to_bytes<QDataType>(M * QK_A)));
    BRIDGE_HIP_CHECK(kFn, BQ_dev.allocate(elements_to_bytes<QDataType>(QK_B * QN_B)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));

    // Host-side B prep: PreshuffleB kernels shuffle B first (shuffle_b_permuteN
    // when TiledMMAPermuteN && kN==1, else shuffle_b); plain copy otherwise.
    if constexpr(SelectedKernel::PreshuffleB)
    {
        auto b_k_n = load_host_tensor<false>(
            B_host, static_cast<int>(K), static_cast<int>(N), static_cast<int>(K));
        constexpr bool use_permute_n = SelectedKernel::TiledMMAPermuteN && (BGroupSizeN == 1);
        auto b_shuffled              = [&]() {
            if constexpr(use_permute_n)
                return ck_tile::shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>(b_k_n);
            else
                return ck_tile::shuffle_b<typename SelectedKernel::BShuffleConfig>(b_k_n);
        }();
        BRIDGE_HIP_CHECK(kFn,
                         hipMemcpy(B_dev,
                                   b_shuffled.data(),
                                   elements_to_bytes<BDataType>(K * N),
                                   hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }

    // Host-side AQ prep (shuffle_aq when APreshuffleQuant, else plain) and BQ prep
    // (bq_permuteN / shuffle_bq / plain copy); both shared verbatim with the
    // aquant and bquant bridges -- see prepare_aq_device() / prepare_bq_device().
    BRIDGE_HIP_CHECK(
        kFn, (prepare_aq_device<SelectedKernel, AQuantGroupSize::kK>(AQ_host, AQ_dev, M, QK_A)));
    BRIDGE_HIP_CHECK(kFn,
                     (prepare_bq_device<SelectedKernel, BQuantGroupSize::kK, BGroupSizeN>(
                         BQ_host, BQ_dev, QK_B, QN_B)));
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // ABQuant: both aq_ptr and bq_ptr are non-null.
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = static_cast<ck_tile::index_t>(QK_B);
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = static_cast<ck_tile::index_t>(stride_BQ);

    return launch_and_copyback<SelectedKernel, CDataType>(
        kFn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // extern "C"
