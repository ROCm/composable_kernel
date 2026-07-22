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

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType, AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// GPU architecture is derived from the running device at launch time (see the
// runtime check in dispatcher_run_bquant_gemm) rather than assumed at compile
// time -- do not hardcode a default architecture here.

static bool g_initialized = false;

#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            return -1;                                                                         \
        }                                                                                      \
    }

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_bquant_gemm.
 *
 * This library uses a single-kernel-per-.so model: SelectedKernel is
 * force-included at compile time and invoked directly via SelectedKernel::launch().
 * No dispatcher registry is involved -- BQuant kernels require QuantGemmHostArgs
 * which is incompatible with the GeneratedTileKernelInstance::run() signature that
 * the dispatcher's registry backend uses.
 *
 * Returns 0 on success.
 */
int dispatcher_initialize()
{
    if(g_initialized)
        return 0;
    g_initialized = true;
    return 0;
}

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
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_bquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !BQ || !C)
    {
        std::cerr << "dispatcher_run_bquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_B <= 0 || QN_B <= 0)
    {
        std::cerr << "dispatcher_run_bquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_bquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_bquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx90a, gfx942, gfx950)\n";
            return -1;
        }
    }

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

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    BDataType* B_dev  = nullptr;
    QDataType* BQ_dev = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(BQ_dev)
            (void)hipFree(BQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // B may be a packed type (pk_int4_t, pk_fp4_t): 2 logical values per byte.
    // elements_to_bytes<T>(n) handles the packed case via numeric_traits::PackedSize.
    if(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&BQ_dev, elements_to_bytes<QDataType>(QK_B * QN_B)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&C_dev, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Copy inputs to device
    if(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup();
        return -1;
    }
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
        if(hipMemcpy(BQ_dev,
                     bq_shuffled.data(),
                     elements_to_bytes<QDataType>(QK_B * QN_B),
                     hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    else
    {
        if(hipMemcpy(
               BQ_dev, BQ_host, elements_to_bytes<QDataType>(QK_B * QN_B), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

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

    const bool do_time = (time_ms != nullptr);
    // When timing is requested use GPU timer with warmup (cold_niters=3, nrepeat=10).
    // Otherwise run once with no overhead.
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
        std::cerr << "dispatcher_run_bquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    if(hipMemcpy(C_host, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) !=
       hipSuccess)
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
 * Initialize dispatcher (alias kept for consistency with gemm_ctypes_lib).
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of kernels in this .so (always 1: the force-included SelectedKernel).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Release resources.
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
