// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * AQuant (A-only quantized) GEMM ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType, QuantGroupSize.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * The *A* matrix is the quantized operand. AQ has shape [M, QK_A] (QK_A =
 * ceil(K/gK)); aq_ptr is set and bq_ptr is nullptr. Its leading dimension
 * follows AQLayout, which is always RowMajor (Old-TE hardcodes AQLayout=RowMajor
 * for every layout), so stride_AQ=QK_A for all layouts. For pk_int4 A the
 * raw values are permuted (permute_i4_inplace) before the device copy, and
 * APreshuffleQuant kernels shuffle AQ via shuffle_aq (row-major only; ccr is
 * excluded from the preshufflequant path by Old-TE).
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
 * Run AQuantGrouped GEMM: C[M,N] = dequant(A[M,K], AQ[M, ceil(K/gK)]) @ B[K,N].
 * A, AQ, B, C are host pointers; device memory is managed internally. QK_A is
 * the number of K-groups = ceil(K / quant_group_k). time_ms is optional.
 * Returns 0 on success.
 */
int dispatcher_run_aquant_gemm(const void* A,
                               const void* AQ,
                               const void* B,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t K,
                               int64_t stride_A,
                               int64_t stride_AQ,
                               int64_t stride_B,
                               int64_t stride_C,
                               int64_t QK_A,
                               int k_batch,
                               float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_aquant_gemm";

    if(!check_entry_args(kFn,
                         g_initialized,
                         {A, AQ, B, C},
                         {M, N, K, QK_A},
                         /*allow_gfx90a=*/true))
        return -1;

    // Validate QK_A against the compile-time quant group size baked into this .so.
    if(!check_quant_group_count(kFn, "QK_A", QK_A, "K", K, QuantGroupSize::kK))
        return -1;

    // Only packed layouts are supported; expected leading dims depend on the
    // compile-time A/B/AQ layouts. AQLayout is always RowMajor (matches Old-TE), so
    // aq_row is true and stride_AQ=QK_A for every layout, including ccr.
    constexpr bool a_row  = std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>;
    constexpr bool b_row  = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
    constexpr bool aq_row = std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>;
    {
        const int64_t exp_stride_A  = a_row ? K : M;
        const int64_t exp_stride_B  = b_row ? N : K;
        const int64_t exp_stride_AQ = aq_row ? QK_A : M;
        const int64_t exp_stride_C  = N;
        if(stride_A != exp_stride_A || stride_B != exp_stride_B || stride_AQ != exp_stride_AQ ||
           stride_C != exp_stride_C)
        {
            std::cerr << kFn << ": non-packed strides are not supported. Expected stride_A="
                      << exp_stride_A << " stride_AQ=" << exp_stride_AQ
                      << " stride_B=" << exp_stride_B << " stride_C=" << exp_stride_C
                      << ", got stride_A=" << stride_A << " stride_AQ=" << stride_AQ
                      << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
            return -1;
        }
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);

    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<QDataType> AQ_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, AQ_dev.allocate(elements_to_bytes<QDataType>(M * QK_A)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    // Copy A. For pk_int4 A the raw i4x4 values must be permuted for the device
    // implementation (run_gemm_quant_example.inc:758-763).
    if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
    {
        auto a_h = load_host_tensor<a_row>(
            A_host, static_cast<int>(M), static_cast<int>(K), static_cast<int>(stride_A));
        permute_i4_inplace(a_h);
        BRIDGE_HIP_CHECK(
            kFn,
            hipMemcpy(
                A_dev, a_h.data(), elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    }

    // Apply AQ preshuffle when required; shared with abquant, see
    // prepare_aq_device() (run_gemm_quant_example.inc:746-751). The assert is
    // AQuant-specific: shuffle_aq assumes a row-major AQ descriptor, which holds
    // here because Old-TE rejects the ccr layout for the preshufflequant path.
    // (abquant's AQ *can* be column-major, but never with APreshuffleQuant on.)
    static_assert(!SelectedKernel::APreshuffleQuant ||
                      std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>,
                  "APreshuffleQuant requires a row-major AQ layout (ccr is excluded "
                  "from the preshufflequant path); shuffle_aq assumes row-major");
    BRIDGE_HIP_CHECK(
        kFn, (prepare_aq_device<SelectedKernel, QuantGroupSize::kK>(AQ_host, AQ_dev, M, QK_A)));
    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // AQuant-only: bq_ptr = nullptr, QK_B = 0, stride_BQ = 0.
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = nullptr;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = 0;
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = 0;

    return launch_and_copyback<SelectedKernel, CDataType>(
        kFn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // extern "C"
