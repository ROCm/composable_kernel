// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GEMM Multi-ABD Dispatcher ctypes Library
 *
 * Provides a C API for Python ctypes integration for the gemm_multi_abd op.
 *
 * WHY A SEPARATE .so (divergent ABI):
 *   gemm_multi_abd is a multi-tensor op: it takes ARRAYS of A, B and D device
 *   pointers (NumATensors / NumBTensors / NumDTensors), not the single A/B/C
 *   triple the regular GEMM ABI (dispatcher_run_gemm) exposes. Its launch takes
 *   a ck_tile::GemmMultiABDHostArgs<NumA, NumB, NumD>, so this library bypasses
 *   the name-keyed dispatcher registry and calls SelectedKernel::launch(...)
 *   directly on the force-included kernel -- exactly the divergent-ABI bridge
 *   pattern used for grouped GEMM (#9000). The kernel header is force-included
 *   via the -include compiler flag and defines SelectedKernel, KERNEL_NAME, and
 *   the NumA/NumB/NumD tensor counts (inside the kernel's namespace, re-exported
 *   to global scope under CK_TILE_SINGLE_KERNEL_INCLUDE).
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libgemm_multi_abd_<name>.so")
 *   lib.dispatcher_initialize()
 *   lib.dispatcher_run_multi_abd(as_ptrs, bs_ptrs, ds_ptrs, e_ptr,
 *                                num_a, num_b, num_d, M, N, K, &time_ms)
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"

// Kernel header force-included via -include compiler flag.
// Defines (under CK_TILE_SINGLE_KERNEL_INCLUDE):
//   SelectedKernel  -- the generated Kernel_<name> struct with a static
//                      launch(const GemmMultiABDHostArgs<...>&, stream_config&)
//   KERNEL_NAME     -- the byte-exact runtime kernel name
//   NumATensors / NumBTensors / NumDTensors -- tensor counts baked into the type

// GPU architecture - must be supplied at compile time via -DGFX_ARCH=<arch>.
#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

// The force-included header exports the tensor counts into the global namespace
// under CK_TILE_SINGLE_KERNEL_INCLUDE. Guard with a fallback so the file is
// still self-describing if the macros ever change.
#ifndef GEMM_MULTI_ABD_NUM_A
#define GEMM_MULTI_ABD_NUM_A NumATensors
#endif
#ifndef GEMM_MULTI_ABD_NUM_B
#define GEMM_MULTI_ABD_NUM_B NumBTensors
#endif
#ifndef GEMM_MULTI_ABD_NUM_D
#define GEMM_MULTI_ABD_NUM_D NumDTensors
#endif

namespace {

constexpr ck_tile::index_t kNumA = GEMM_MULTI_ABD_NUM_A;
constexpr ck_tile::index_t kNumB = GEMM_MULTI_ABD_NUM_B;
constexpr ck_tile::index_t kNumD = GEMM_MULTI_ABD_NUM_D;

using HostArgs = ck_tile::GemmMultiABDHostArgs<kNumA, kNumB, kNumD>;

bool g_initialized = false;

} // namespace

extern "C" {

/**
 * Initialize the library. Multi-ABD is registry-bypass (it calls
 * SelectedKernel::launch directly), so there is no registry to populate here;
 * this just flips the ready flag and is kept for ABI symmetry with the regular
 * GEMM ctypes lib.
 *
 * Returns: 0 on success.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of tensors the force-included kernel expects (compile-time constants).
 */
int dispatcher_get_num_a_tensors() { return static_cast<int>(kNumA); }
int dispatcher_get_num_b_tensors() { return static_cast<int>(kNumB); }
int dispatcher_get_num_d_tensors() { return static_cast<int>(kNumD); }

/**
 * Kernel name (single-kernel ABI): the byte-exact KERNEL_NAME baked into the
 * force-included header. Mirrors dispatcher_get_kernel_name in the regular lib.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Multi-kernel ABI shim: this .so holds exactly one force-included kernel, so
 * index 0 returns KERNEL_NAME and any other index is out of range. Kept so the
 * Python GemmDispatcherLib multi-kernel path works uniformly across variants.
 */
int dispatcher_get_kernel_name_at(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0 || index != 0)
    {
        return -1;
    }
    std::strncpy(buffer, KERNEL_NAME, static_cast<size_t>(buffer_size) - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

int dispatcher_get_kernel_count() { return 1; }

/**
 * Run a multi-ABD GEMM on the GPU via the force-included kernel.
 *
 * The Python runner hands HOST buffers (contiguous, already laid out for the
 * kernel's compiled layout); this shim owns the GPU side -- it hipMallocs one
 * device buffer per A/B/D tensor and for E, copies inputs up, launches, and
 * copies the result back. This mirrors the regular gemm_ctypes_lib's
 * host-pointer contract so GpuGemmRunner-style callers just pass numpy arrays.
 *
 * as_hosts / bs_hosts / ds_hosts : arrays of host pointers (length num_a/b/d).
 * e_host                         : output host pointer (M*N * sizeof(EDataType)).
 * elem_a/b/d/e                   : element size in bytes for each group's dtype
 *                                  (so this shim need not know the CK dtype).
 * stride_*                       : per-operand leading stride. These are forwarded
 *                                  verbatim into GemmMultiABDHostArgs; ck_tile's
 *                                  UniversalGemmKernel uses them AS-IS and does NOT
 *                                  treat 0 as a "derive the default" sentinel (a 0
 *                                  or negative stride collapses the tensor
 *                                  descriptor and corrupts addressing). They MUST
 *                                  therefore be non-null and strictly positive;
 *                                  the caller computes them from M,N,K and the
 *                                  compiled layout (see GpuMultiABDRunner).
 * time_ms                        : filled with the average kernel time (ms).
 *
 * Returns 0 on success, negative on error. num_a/num_b/num_d MUST equal the
 * kernel's compiled tensor counts or the call is rejected (-3). Missing or
 * non-positive strides are rejected (-1).
 */
int dispatcher_run_multi_abd(const void** as_hosts,
                             const void** bs_hosts,
                             const void** ds_hosts,
                             void* e_host,
                             const int64_t* stride_as,
                             const int64_t* stride_bs,
                             const int64_t* stride_ds,
                             int64_t stride_e,
                             int elem_a,
                             int elem_b,
                             int elem_d,
                             int elem_e,
                             int num_a,
                             int num_b,
                             int num_d,
                             int64_t M,
                             int64_t N,
                             int64_t K,
                             float* time_ms)
{
    if(!g_initialized || !as_hosts || !bs_hosts || !e_host)
    {
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || elem_a <= 0 || elem_b <= 0 || elem_e <= 0)
    {
        return -1;
    }
    // The tensor counts are baked into the kernel type at compile time; a
    // mismatch would silently read past the caller's arrays, so reject it.
    if(num_a != static_cast<int>(kNumA) || num_b != static_cast<int>(kNumB) ||
       num_d != static_cast<int>(kNumD))
    {
        return -3;
    }
    if(kNumD > 0 && (!ds_hosts || elem_d <= 0))
    {
        return -1;
    }
    // Strides are forwarded verbatim to the kernel (no default derivation), so a
    // null array or any non-positive value would silently corrupt addressing.
    // Require explicit, strictly-positive strides for every operand.
    if(!stride_as || !stride_bs || stride_e <= 0 || (kNumD > 0 && !stride_ds))
    {
        return -1;
    }
    for(int i = 0; i < num_a; ++i)
    {
        if(stride_as[i] <= 0)
        {
            return -1;
        }
    }
    for(int i = 0; i < num_b; ++i)
    {
        if(stride_bs[i] <= 0)
        {
            return -1;
        }
    }
    for(int i = 0; i < num_d; ++i)
    {
        if(stride_ds[i] <= 0)
        {
            return -1;
        }
    }

    // Cast every factor to size_t so the products are computed in 64-bit
    // unsigned arithmetic (no reliance on operand-promotion order).
    const size_t a_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(K) * static_cast<size_t>(elem_a);
    const size_t b_bytes =
        static_cast<size_t>(K) * static_cast<size_t>(N) * static_cast<size_t>(elem_b);
    const size_t d_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(elem_d);
    const size_t e_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(elem_e);

    std::vector<void*> a_dev(kNumA, nullptr), b_dev(kNumB, nullptr), d_dev(kNumD, nullptr);
    void* e_dev = nullptr;

    auto cleanup = [&]() {
        for(auto p : a_dev)
            if(p)
                (void)hipFree(p);
        for(auto p : b_dev)
            if(p)
                (void)hipFree(p);
        for(auto p : d_dev)
            if(p)
                (void)hipFree(p);
        if(e_dev)
            (void)hipFree(e_dev);
    };

    // Allocate + upload each operand tensor.
    for(int i = 0; i < num_a; ++i)
    {
        if(hipMalloc(&a_dev[i], a_bytes) != hipSuccess)
        {
            cleanup();
            return -1;
        }
        if(hipMemcpy(a_dev[i], as_hosts[i], a_bytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    for(int i = 0; i < num_b; ++i)
    {
        if(hipMalloc(&b_dev[i], b_bytes) != hipSuccess)
        {
            cleanup();
            return -1;
        }
        if(hipMemcpy(b_dev[i], bs_hosts[i], b_bytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    for(int i = 0; i < num_d; ++i)
    {
        if(hipMalloc(&d_dev[i], d_bytes) != hipSuccess)
        {
            cleanup();
            return -1;
        }
        if(hipMemcpy(d_dev[i], ds_hosts[i], d_bytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    if(hipMalloc(&e_dev, e_bytes) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(e_dev, 0, e_bytes) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Pack the device pointers / strides into the std::array shapes the
    // GemmMultiABDHostArgs constructor requires.
    std::array<const void*, kNumA> as{};
    std::array<const void*, kNumB> bs{};
    std::array<const void*, kNumD> ds{};
    std::array<ck_tile::index_t, kNumA> str_as{};
    std::array<ck_tile::index_t, kNumB> str_bs{};
    std::array<ck_tile::index_t, kNumD> str_ds{};

    // ck_tile::index_t is 32-bit: reject any dimension or stride that would not
    // fit before the static_casts below, so an out-of-range value fails loudly
    // instead of silently truncating.
    {
        auto fits_i32 = [](int64_t v) { return v <= static_cast<int64_t>(INT32_MAX); };
        if(!fits_i32(M) || !fits_i32(N) || !fits_i32(K) || !fits_i32(stride_e))
        {
            std::cerr << "dispatcher_run_multi_abd: M/N/K or stride_e exceeds the 32-bit "
                         "index range\n";
            return -1;
        }
        for(ck_tile::index_t i = 0; i < kNumA; ++i)
            if(!fits_i32(stride_as[i]))
            {
                std::cerr << "dispatcher_run_multi_abd: stride_as exceeds the 32-bit index "
                             "range\n";
                return -1;
            }
        for(ck_tile::index_t i = 0; i < kNumB; ++i)
            if(!fits_i32(stride_bs[i]))
            {
                std::cerr << "dispatcher_run_multi_abd: stride_bs exceeds the 32-bit index "
                             "range\n";
                return -1;
            }
        for(ck_tile::index_t i = 0; i < kNumD; ++i)
            if(!fits_i32(stride_ds[i]))
            {
                std::cerr << "dispatcher_run_multi_abd: stride_ds exceeds the 32-bit index "
                             "range\n";
                return -1;
            }
    }

    // Strides validated non-null, strictly positive, and 32-bit-safe above, so
    // pack directly.
    for(ck_tile::index_t i = 0; i < kNumA; ++i)
    {
        as[i]     = a_dev[i];
        str_as[i] = static_cast<ck_tile::index_t>(stride_as[i]);
    }
    for(ck_tile::index_t i = 0; i < kNumB; ++i)
    {
        bs[i]     = b_dev[i];
        str_bs[i] = static_cast<ck_tile::index_t>(stride_bs[i]);
    }
    for(ck_tile::index_t i = 0; i < kNumD; ++i)
    {
        ds[i]     = d_dev[i];
        str_ds[i] = static_cast<ck_tile::index_t>(stride_ds[i]);
    }

    // Multi-ABD supports only k_batch = 1.
    HostArgs args{as,
                  bs,
                  ds,
                  e_dev,
                  /*k_batch=*/1,
                  static_cast<ck_tile::index_t>(M),
                  static_cast<ck_tile::index_t>(N),
                  static_cast<ck_tile::index_t>(K),
                  str_as,
                  str_bs,
                  str_ds,
                  static_cast<ck_tile::index_t>(stride_e)};

    float exec_time = -1.0f;
    try
    {
        // Registry bypass: launch the force-included kernel directly. The
        // launch() returns a single float average time (unlike some Old-TE
        // launchers that return a tuple), so no tuple normalization is needed.
        ck_tile::stream_config stream{nullptr, /*time_kernel=*/true};
        exec_time = SelectedKernel::launch(args, stream);
    }
    catch(const std::exception& e)
    {
        std::cerr << "gemm_multi_abd launch failed: " << e.what() << std::endl;
        cleanup();
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2;
    }

    // Copy result back to the host output buffer.
    if(hipMemcpy(e_host, e_dev, e_bytes, hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    cleanup();
    if(time_ms)
    {
        *time_ms = exec_time;
    }
    return 0;
}

void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
