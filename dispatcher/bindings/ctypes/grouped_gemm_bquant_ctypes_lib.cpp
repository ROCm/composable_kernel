// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm BQuant ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_bquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType, QuantGroupSize
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: BQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ck_tile/host/tensor_shuffle_utils.hpp"

#include "quant_bridge_common.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType, AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME
//
// Shared infrastructure (elements_to_bytes, DeviceBuffer, validate_supported_arch,
// make_stream_config, launch<>, BRIDGE_HIP_CHECK, QUANT_BRIDGE_C_API) lives in
// quant_bridge_common.hpp. GPU architecture is derived from the running device at
// launch time (see validate_supported_arch) rather than assumed at compile time.

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run BQuantGrouped GEMM: C[M,N] = A[M,K] @ dequant(B[K,N], BQ[ceil(K/gK), ceil(N/gN)])
 *
 * A, B, BQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, BQ, C  - host data pointers
 *   M, N, K      - matrix dimensions
 *   stride_A     - leading dimension of A (row-major: K; col-major: M)
 *   stride_B     - leading dimension of B (col-major: K; row-major: N)
 *   stride_BQ    - leading dimension of BQ (row-major: ceil(N/gN))
 *   stride_C     - leading dimension of C (row-major: N)
 *   QK_B         - number of K-groups = ceil(K / quant_group_k)
 *   QN_B         - number of N-groups = ceil(N / quant_group_n)
 *   k_batch      - split-K factor (1 = no split)
 *   time_ms      - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
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

    if(!g_initialized)
    {
        std::cerr << kFn << ": not initialized\n";
        return -1;
    }
    if(!A || !B || !BQ || !C)
    {
        std::cerr << kFn << ": null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_B <= 0 || QN_B <= 0)
    {
        std::cerr << kFn << ": invalid dimensions\n";
        return -1;
    }
    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    if(!validate_supported_arch(kFn, /*allow_gfx90a=*/true))
        return -1;

    // Validate that the caller's QK_B/QN_B match the compile-time quant group sizes
    // baked into this .so.  A mismatch means the BQ device buffer would be allocated
    // with the wrong size while the kernel indexes it with different strides.
    {
        const int64_t expected_QK_B =
            (K + static_cast<int64_t>(QuantGroupSize::kK) - 1) / QuantGroupSize::kK;
        const int64_t expected_QN_B =
            (N + static_cast<int64_t>(QuantGroupSize::kN) - 1) / QuantGroupSize::kN;
        if(QK_B != expected_QK_B || QN_B != expected_QN_B)
        {
            std::cerr << "dispatcher_run_bquant_gemm: QK_B/QN_B mismatch. " << "Got (" << QK_B
                      << ", " << QN_B << "), " << "expected (" << expected_QK_B << ", "
                      << expected_QN_B << ") " << "for K=" << K << " N=" << N
                      << " with QuantGroupSize kK=" << QuantGroupSize::kK
                      << " kN=" << QuantGroupSize::kN << "\n";
            return -1;
        }
    }

    // This implementation only supports packed (contiguous) layouts.
    // Device buffers are allocated and copied as M*K, K*N, QK_B*QN_B, M*N packed arrays.
    // Non-packed strides would cause the kernel to index into a differently-sized buffer,
    // producing incorrect results or out-of-bounds accesses.
    if(stride_A != K || stride_B != K || stride_BQ != QN_B || stride_C != N)
    {
        std::cerr << "dispatcher_run_bquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_BQ=" << QN_B
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_BQ=" << stride_BQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);

    // RAII device buffers: any early return (including from BRIDGE_HIP_CHECK) frees
    // every allocation automatically -- no hand-written cleanup lambda needed.
    // B may be a packed type (pk_int4_t, pk_fp4_t): 2 logical values per byte.
    // elements_to_bytes<T>(n) handles the packed case via numeric_traits::PackedSize.
    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> BQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, BQ_dev.allocate(elements_to_bytes<QDataType>(QK_B * QN_B)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    // Copy inputs to device
    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    // Apply BQ preshuffle when required -- mirrors gemm_bquant_profiler.hpp:118-121.
    // BPreshuffleQuant reorders BQ in host memory before the device copy so the kernel
    // finds the scale values in the interleaved layout it expects.
    if constexpr(SelectedKernel::BPreshuffleQuant)
    {
        constexpr int block_bq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(QuantGroupSize::kK);
        ck_tile::HostTensor<QDataType> bq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QK_B),
                                            static_cast<int>(QN_B),
                                            static_cast<int>(QN_B),
                                            ck_tile::bool_constant<true>{} /*row-major*/));
        std::copy(BQ_host, BQ_host + QK_B * QN_B, bq_h.begin());
        auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
        BRIDGE_HIP_CHECK(kFn,
                         hipMemcpy(BQ_dev,
                                   bq_shuffled.data(),
                                   elements_to_bytes<QDataType>(QK_B * QN_B),
                                   hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn,
            hipMemcpy(
                BQ_dev, BQ, elements_to_bytes<QDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // Build QuantGemmHostArgs (aq_ptr = nullptr, QK_A = 0, stride_AQ = 0 for BQuant-only)
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

    // When timing is requested use GPU timer with warmup (cold_niters=3, nrepeat=10);
    // otherwise run once with no overhead (make_stream_config handles both).
    const float exec_time = launch<SelectedKernel>(args, time_ms != nullptr);
    if(exec_time < 0.0f)
    {
        std::cerr << kFn << ": kernel reported unsupported args\n";
        return -2;
    }

    // Copy result back
    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(C, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost));

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

} // extern "C"
