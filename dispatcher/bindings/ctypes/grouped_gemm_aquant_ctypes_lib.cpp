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

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType (AQDataType), AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// GPU architecture is derived from the running device at launch time rather than
// assumed at compile time -- do not hardcode a default architecture here.

static bool g_initialized = false;
static std::string g_gfx_arch;

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_aquant_gemm.
 *
 * Queries and caches the GPU architecture so subsequent run calls avoid the
 * per-call hipGetDeviceProperties overhead.
 *
 * Returns 0 on success, -1 if device query fails or arch is unsupported.
 */
int dispatcher_initialize()
{
    if(g_initialized)
        return 0;

    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << "dispatcher_initialize: could not query device architecture\n";
        return -1;
    }
    g_gfx_arch = props.gcnArchName;
    if(g_gfx_arch.rfind("gfx950", 0) != 0 && g_gfx_arch.rfind("gfx942", 0) != 0 &&
       g_gfx_arch.rfind("gfx90a", 0) != 0)
    {
        std::cerr << "dispatcher_initialize: unsupported GPU architecture '" << g_gfx_arch
                  << "' (supported: gfx90a, gfx942, gfx950)\n";
        return -1;
    }
    g_initialized = true;
    return 0;
}

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
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_aquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !C)
    {
        std::cerr << "dispatcher_run_aquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_A <= 0 || QM_A <= 0)
    {
        std::cerr << "dispatcher_run_aquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Validate that the caller's QK_A/QM_A match the compile-time quant group sizes.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(QuantGroupSize::kK) - 1) / QuantGroupSize::kK;
        const int64_t expected_QM_A =
            (M + static_cast<int64_t>(QuantGroupSize::kM) - 1) / QuantGroupSize::kM;
        if(QK_A != expected_QK_A || QM_A != expected_QM_A)
        {
            std::cerr << "dispatcher_run_aquant_gemm: QK_A/QM_A mismatch. " << "Got (" << QK_A
                      << ", " << QM_A << "), " << "expected (" << expected_QK_A << ", "
                      << expected_QM_A << ") " << "for K=" << K << " M=" << M
                      << " with QuantGroupSize kK=" << QuantGroupSize::kK
                      << " kM=" << QuantGroupSize::kM << "\n";
            return -1;
        }
    }

    // Only packed (contiguous) layouts are supported.
    // AQ is RowMajor [QM_A, QK_A]: stride_AQ == QK_A
    if(stride_A != K || stride_B != K || stride_AQ != QK_A || stride_C != N)
    {
        std::cerr << "dispatcher_run_aquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_AQ=" << QK_A
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_AQ=" << stride_AQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    BDataType* B_dev  = nullptr;
    QDataType* AQ_dev = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(AQ_dev)
            (void)hipFree(AQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // B may be a packed type (pk_int4_t): elements_to_bytes<T>(n) handles PackedSize correctly.
    if(hipMalloc(&A_dev,
                 elements_to_bytes<ADataType>(static_cast<std::size_t>(M) *
                                              static_cast<std::size_t>(K))) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev,
                 elements_to_bytes<BDataType>(static_cast<std::size_t>(K) *
                                              static_cast<std::size_t>(N))) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&AQ_dev, elements_to_bytes<QDataType>(QM_A * QK_A)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&C_dev,
                 elements_to_bytes<CDataType>(static_cast<std::size_t>(M) *
                                              static_cast<std::size_t>(N))) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Copy inputs to device
    if(hipMemcpy(
           A_dev,
           A_host,
           elements_to_bytes<ADataType>(static_cast<std::size_t>(M) * static_cast<std::size_t>(K)),
           hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(
           B_dev,
           B_host,
           elements_to_bytes<BDataType>(static_cast<std::size_t>(K) * static_cast<std::size_t>(N)),
           hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
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
        if(hipMemcpy(AQ_dev,
                     aq_shuffled.data(),
                     elements_to_bytes<QDataType>(QM_A * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    else
    {
        if(hipMemcpy(
               AQ_dev, AQ_host, elements_to_bytes<QDataType>(QM_A * QK_A), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    if(hipMemset(C_dev,
                 0,
                 elements_to_bytes<CDataType>(static_cast<std::size_t>(M) *
                                              static_cast<std::size_t>(N))) != hipSuccess)
    {
        cleanup();
        return -1;
    }

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

    const bool do_time = (time_ms != nullptr);
    ck_tile::stream_config stream_cfg{
        nullptr,          // stream_id_
        do_time,          // time_kernel_
        0,                // log_level_
        do_time ? 3 : 0,  // cold_niters_
        do_time ? 10 : 1, // nrepeat_
        do_time,          // is_gpu_timer_
        false,            // flush_cache_
        1,                // rotating_count_
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_aquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    if(hipMemcpy(
           C_host,
           C_dev,
           elements_to_bytes<CDataType>(static_cast<std::size_t>(M) * static_cast<std::size_t>(N)),
           hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

/**
 * Return the compile-time KERNEL_NAME of the force-included kernel.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Initialize dispatcher (alias kept for consistency).
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of kernels registered (always 1 for the single-kernel-per-.so model).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Release dispatcher resources.
 */
void dispatcher_cleanup()
{
    g_initialized = false;
    g_gfx_arch.clear();
}

} // extern "C"
