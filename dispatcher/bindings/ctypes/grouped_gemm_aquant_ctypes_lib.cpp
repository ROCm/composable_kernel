// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm AQuant Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Kernel header is force-included at compile time via:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_aquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType (=AQDataType), AccDataType, QuantGroupSize
 *
 * AQuant: A-side activation quantization.
 *   AQ[ceil(M/gM), ceil(K/gK)] is the A-side scale tensor (RowMajor).
 *   BQ is unused (bq_ptr=nullptr, QK_B=0, stride_BQ=0 in QuantGemmHostArgs).
 *
 * APreshuffleQuant: when SelectedKernel::APreshuffleQuant==true (pipeline=compv3),
 *   AQ is reordered in host memory via shuffle_aq() before the device copy, matching
 *   the interleaved layout the kernel's block tile expects.
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: AQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include <cstdint>
#include <iostream>
#include <string>

#include "quant_bridge_common.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType (AQDataType), AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME
//
// Shared infrastructure (elements_to_bytes, DeviceBuffer, validate_supported_arch,
// make_stream_config, launch<>, BRIDGE_HIP_CHECK, QUANT_BRIDGE_C_API) lives in
// quant_bridge_common.hpp. GPU architecture is derived from the running device at
// launch time (see validate_supported_arch) rather than assumed at compile time.

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run AQuantGrouped GEMM: C[M,N] = dequant(A[M,K], AQ[ceil(M/gM), ceil(K/gK)]) @ B[K,N]
 *
 * A, B, AQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, AQ, C  - host data pointers
 *   M, N, K      - matrix dimensions
 *   stride_A     - leading dimension of A (row-major: K)
 *   stride_B     - leading dimension of B (col-major: K)
 *   stride_AQ    - leading dimension of AQ (row-major: ceil(K/gK))
 *   stride_C     - leading dimension of C (row-major: N)
 *   QK_A         - number of K-groups for A = ceil(K / quant_group_k)
 *   QM_A         - number of M-groups for A = ceil(M / quant_group_m)  (typically == M when gM=1)
 *   k_batch      - split-K factor (1 = no split)
 *   time_ms      - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_aquant_gemm(const void* A,
                               const void* B,
                               const void* AQ,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t K,
                               int64_t stride_A,
                               int64_t stride_B,
                               int64_t stride_AQ,
                               int64_t stride_C,
                               int64_t QK_A,
                               int64_t QM_A,
                               int k_batch,
                               float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_aquant_gemm";

    if(!g_initialized)
    {
        std::cerr << kFn << ": not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !C)
    {
        std::cerr << kFn << ": null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_A <= 0 || QM_A <= 0)
    {
        std::cerr << kFn << ": invalid dimensions\n";
        return -1;
    }
    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs.
    if(!validate_supported_arch(kFn, /*allow_gfx90a=*/true))
        return -1;

    // Validate that the caller's QK_A/QM_A match the compile-time quant group sizes.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(QuantGroupSize::kK) - 1) / QuantGroupSize::kK;
        const int64_t expected_QM_A =
            (M + static_cast<int64_t>(QuantGroupSize::kM) - 1) / QuantGroupSize::kM;
        if(QK_A != expected_QK_A || QM_A != expected_QM_A)
        {
            std::cerr << kFn << ": QK_A/QM_A mismatch. " << "Got (" << QK_A << ", " << QM_A << "), "
                      << "expected (" << expected_QK_A << ", " << expected_QM_A << ") "
                      << "for K=" << K << " M=" << M
                      << " with QuantGroupSize kK=" << QuantGroupSize::kK
                      << " kM=" << QuantGroupSize::kM << "\n";
            return -1;
        }
    }

    // Only packed (contiguous) layouts are supported.
    // AQ is RowMajor [QM_A, QK_A]: stride_AQ == QK_A
    if(stride_A != K || stride_B != K || stride_AQ != QK_A || stride_C != N)
    {
        std::cerr << kFn << ": non-packed strides are not supported. " << "Expected stride_A=" << K
                  << " stride_B=" << K << " stride_AQ=" << QK_A << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_AQ=" << stride_AQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);

    // RAII device buffers: any early return (including from BRIDGE_HIP_CHECK) frees
    // every allocation automatically -- no hand-written cleanup lambda needed.
    // B may be a packed type (pk_int4_t): elements_to_bytes<T>(n) handles PackedSize correctly.
    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> AQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn,
                     A_dev.allocate(elements_to_bytes<ADataType>(static_cast<std::size_t>(M) *
                                                                 static_cast<std::size_t>(K))));
    BRIDGE_HIP_CHECK(kFn,
                     B_dev.allocate(elements_to_bytes<BDataType>(static_cast<std::size_t>(K) *
                                                                 static_cast<std::size_t>(N))));
    BRIDGE_HIP_CHECK(kFn, AQ_dev.allocate(elements_to_bytes<QDataType>(QM_A * QK_A)));
    BRIDGE_HIP_CHECK(kFn,
                     C_dev.allocate(elements_to_bytes<CDataType>(static_cast<std::size_t>(M) *
                                                                 static_cast<std::size_t>(N))));

    // Copy inputs to device
    BRIDGE_HIP_CHECK(kFn,
                     hipMemcpy(A_dev,
                               A,
                               elements_to_bytes<ADataType>(static_cast<std::size_t>(M) *
                                                            static_cast<std::size_t>(K)),
                               hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(kFn,
                     hipMemcpy(B_dev,
                               B,
                               elements_to_bytes<BDataType>(static_cast<std::size_t>(K) *
                                                            static_cast<std::size_t>(N)),
                               hipMemcpyHostToDevice));
    // Copy AQ to device, preshuffling when the kernel requires it.
    // APreshuffleQuant=true (pipeline=compv3): AQ is reordered so the kernel's block tile
    // finds scale values in the interleaved layout it expects.
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        constexpr int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(QuantGroupSize::kK);
        ck_tile::HostTensor<QDataType> aq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QM_A),
                                            static_cast<int>(QK_A),
                                            static_cast<int>(QK_A),
                                            ck_tile::bool_constant<true>{} /*row-major*/));
        std::copy(AQ_host, AQ_host + QM_A * QK_A, aq_h.begin());
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        BRIDGE_HIP_CHECK(kFn,
                         hipMemcpy(AQ_dev,
                                   aq_shuffled.data(),
                                   elements_to_bytes<QDataType>(QM_A * QK_A),
                                   hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn,
            hipMemcpy(
                AQ_dev, AQ, elements_to_bytes<QDataType>(QM_A * QK_A), hipMemcpyHostToDevice));
    }
    BRIDGE_HIP_CHECK(kFn,
                     hipMemset(C_dev,
                               0,
                               elements_to_bytes<CDataType>(static_cast<std::size_t>(M) *
                                                            static_cast<std::size_t>(N))));

    // Build QuantGemmHostArgs: AQ active, BQ unused
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;  // A-side scale: active for AQuant
    args.bq_ptr    = nullptr; // B-side scale: unused for AQuant-only
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = 0; // unused
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = 0; // unused

    // When timing is requested use GPU timer with warmup (cold_niters=3, nrepeat=10);
    // otherwise run once with no overhead (make_stream_config handles both).
    const float exec_time = launch<SelectedKernel>(args, time_ms != nullptr);
    if(exec_time < 0.0f)
    {
        std::cerr << kFn << ": kernel reported unsupported args\n";
        return -2;
    }

    // Copy result back
    BRIDGE_HIP_CHECK(kFn,
                     hipMemcpy(C,
                               C_dev,
                               elements_to_bytes<CDataType>(static_cast<std::size_t>(M) *
                                                            static_cast<std::size_t>(N)),
                               hipMemcpyDeviceToHost));

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

} // extern "C"
