// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Batched GEMM Dispatcher ctypes Library
 *
 * Provides a C API for Python ctypes integration for the BATCHED GEMM bridge.
 *
 * Unlike gemm_ctypes_lib.cpp (single-problem GEMM), batched GEMM has a
 * divergent ABI: it carries a batch dimension with per-batch strides. The
 * registry / Dispatcher::run() path only knows the single-problem
 * (A, B, C, M, N, K) signature, so this library BYPASSES the registry and
 * launches the force-included kernel directly via
 * ``SelectedKernel::launch(ck_tile::BatchedGemmHostArgs{...}, stream)`` --
 * the same launch entry the Tile Engine batched_gemm benchmark uses. This
 * mirrors the registry-bypass pattern used by the stream-K bridge.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libbatched_gemm_....so")
 *   lib.dispatcher_initialize()   // the name the Python wrapper binds
 *   lib.dispatcher_run_batched(A, B, C, M, N, K, batch_count, k_batch,
 *                              stride_A, stride_B, stride_C,
 *                              batch_stride_A, batch_stride_B, batch_stride_C,
 *                              warmup, repeat, flush_cache, rotating_count,
 *                              &time_ms)
 *
 * The ABI threads k_batch (split-K, Old-TE supports it) and the full
 * benchmarking knobs (warmup/repeat/flush_cache/rotating_count) so this bridge
 * drives ``SelectedKernel::launch`` with the SAME ``stream_config`` the Tile
 * Engine batched_gemm profiler uses -- a prerequisite for a fair A/B timing
 * comparison (the earlier version launched with defaults cold=3/repeat=10).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// Kernel header included via -include compiler flag under
// CK_TILE_SINGLE_KERNEL_INCLUDE. Defines: ADataType, BDataType, CDataType,
// AccDataType, ALayout, BLayout, CLayout, SelectedKernel, KERNEL_NAME.

// GPU architecture - MUST be supplied at compile time via -DGFX_ARCH="<arch>".
// Do not default to a specific GPU architecture: the arch is resolved from the
// host (get_arch / rocminfo) and threaded through as a compile flag so the
// kernel is never silently built for the wrong target.
#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

namespace {

// The batched bridge is single-kernel-per-.so: the force-included header fully
// determines the kernel. There is no registry to initialize, but the init entry
// is kept for ABI symmetry with the single-problem GEMM library so the Python
// runner can call initialize() uniformly.
bool g_initialized = false;

// Log a failing HIP call (with hipGetErrorString) before the caller returns an
// error status: a bare -1 with no diagnostic makes a real GPU sweep failure
// impossible to triage. Returns -1 so call sites read `return fail_hip(...)`.
inline int fail_hip(const char* what, hipError_t err)
{
    std::cerr << "dispatcher_run_batched: " << what << " failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
}

// Default (contiguous) stride for a row/col-major operand, matching
// ck_tile::get_default_stride used by the Tile Engine profiler: a value of 0
// means "packed", so derive it from the problem shape.
inline std::int64_t
default_stride(std::int64_t rows, std::int64_t cols, std::int64_t provided, bool row_major)
{
    // Match ck_tile::get_default_stride / the TE profiler exactly: a provided
    // value of <= 0 means "packed" and the leading stride is derived from the
    // shape. (N2: the earlier `> 0` check disagreed with TE's `<= 0` sentinel;
    // they are equivalent for legal strides but this is the exact form.)
    if(provided <= 0)
    {
        return row_major ? cols : rows;
    }
    return provided;
}

// Per-operand layout is emitted by the codegen as single-char strings
// GEMM_KEY_LAYOUT_A/B/C ("r"/"c"). The generated header only exports the data
// types (not the layout aliases) into the global namespace, so we read layout
// from the macros rather than std::is_same_v<ALayout, ...>.
inline bool operand_is_row_major(const char* layout_char) { return layout_char[0] == 'r'; }

#ifdef GEMM_KEY_LAYOUT_A
constexpr const char* kLayoutA = GEMM_KEY_LAYOUT_A;
constexpr const char* kLayoutB = GEMM_KEY_LAYOUT_B;
constexpr const char* kLayoutC = GEMM_KEY_LAYOUT_C;
#else
constexpr const char* kLayoutA = "r";
constexpr const char* kLayoutB = "c";
constexpr const char* kLayoutC = "r";
#endif

} // namespace

extern "C" {

/**
 * Initialize the library. No registry is used for batched GEMM (the kernel is
 * force-included), so this simply flips a flag. Returns 0 on success.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

// Back-compat alias. `dispatcher_initialize` is the canonical entry (the one the
// Python BatchedGemmDispatcherLib binds); `dispatcher_init` is kept only so the
// same symbol name works across every sibling bridge .so that historically
// exported it. New callers should use dispatcher_initialize.
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Report the compile-time kernel name of the force-included batched kernel.
 * The batched bridge is always one kernel per .so.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Multi-kernel ABI shim: the batched .so exposes exactly one kernel, so index 0
 * returns KERNEL_NAME and every other index fails. Mirrors the single-problem
 * library so the shared Python wrapper can query names uniformly.
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
 * Run a batched GEMM on the GPU via the force-included kernel.
 *
 * Takes HOST pointers and manages GPU memory internally (hipMalloc/hipMemcpy/
 * hipFree), matching the single-problem GEMM ABI. The per-batch strides let the
 * caller lay out A/B/C as [batch_count, rows, cols] tensors; a stride argument
 * of 0 falls back to the packed/default stride.
 *
 * Returns: 0 on success; -1 on a HIP error or bad/guarded arguments (incl. an
 * out-of-range 32-bit index); -2 if the kernel launch throws.
 */
int dispatcher_run_batched(const void* A,
                           const void* B,
                           void* C,
                           std::int64_t M,
                           std::int64_t N,
                           std::int64_t K,
                           std::int64_t batch_count,
                           std::int64_t k_batch,
                           std::int64_t stride_A,
                           std::int64_t stride_B,
                           std::int64_t stride_C,
                           std::int64_t batch_stride_A,
                           std::int64_t batch_stride_B,
                           std::int64_t batch_stride_C,
                           std::int64_t warmup,
                           std::int64_t repeat,
                           std::int64_t flush_cache,
                           std::int64_t rotating_count,
                           float* time_ms)
{
    if(!g_initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    {
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -1;
    }

    // split-K count: <= 0 falls back to 1 (no split), matching the TE default.
    const ck_tile::index_t kbatch = static_cast<ck_tile::index_t>(k_batch > 0 ? k_batch : 1);

    // Resolve leading strides: <= 0 -> packed default (matches TE profiler /
    // ck_tile::get_default_stride behaviour), respecting each operand's own
    // row/col-major layout. For rcr this yields stride_A=K, stride_B=K,
    // stride_C=N.
    const std::int64_t sa = default_stride(M, K, stride_A, operand_is_row_major(kLayoutA));
    const std::int64_t sb = default_stride(K, N, stride_B, operand_is_row_major(kLayoutB));
    const std::int64_t sc = default_stride(M, N, stride_C, operand_is_row_major(kLayoutC));

    // Batch strides: <= 0 -> packed default (contiguous per-batch slab). The
    // packed slab spans the RESOLVED leading stride (sa/sb/sc), not the bare
    // shape, so a non-packed leading stride (padded rows/cols) is honoured: a
    // row-major operand's slab is rows*leading and a col-major operand's slab is
    // cols*leading. Deriving the default from M*K/K*N/M*N would under-size the
    // buffer and cause out-of-bounds access whenever stride_A/B/C is padded. A
    // caller may still pass an even LARGER batch stride (extra inter-slab
    // padding); the element-count / allocation below is sized from the resolved
    // value so that layout is honoured too.
    const std::int64_t bsa =
        batch_stride_A > 0 ? batch_stride_A : (operand_is_row_major(kLayoutA) ? M : K) * sa;
    const std::int64_t bsb =
        batch_stride_B > 0 ? batch_stride_B : (operand_is_row_major(kLayoutB) ? K : N) * sb;
    const std::int64_t bsc =
        batch_stride_C > 0 ? batch_stride_C : (operand_is_row_major(kLayoutC) ? M : N) * sc;

    // Total element counts across all batches, sized from the RESOLVED batch
    // stride so that a padded (non-packed) batch stride allocates enough for the
    // last batch's slab plus its padding.
    const std::int64_t a_elems = bsa * batch_count;
    const std::int64_t b_elems = bsb * batch_count;
    const std::int64_t c_elems = bsc * batch_count;

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // ck_tile::index_t is 32-bit: reject any dimension / stride / batch value
    // that would not fit before the static_casts into BatchedGemmHostArgs below,
    // so an out-of-range problem fails loudly instead of silently truncating.
    {
        auto fits_i32 = [](std::int64_t v) { return v <= static_cast<std::int64_t>(INT32_MAX); };
        if(!fits_i32(M) || !fits_i32(N) || !fits_i32(K) || !fits_i32(sa) || !fits_i32(sb) ||
           !fits_i32(sc) || !fits_i32(bsa) || !fits_i32(bsb) || !fits_i32(bsc) ||
           !fits_i32(batch_count) || !fits_i32(kbatch))
        {
            std::cerr << "dispatcher_run_batched: a dimension/stride/batch value exceeds the "
                         "32-bit index range\n";
            if(time_ms)
            {
                *time_ms = -1.0f;
            }
            return -1;
        }
    }

    hipError_t err;
    if((err = hipMalloc(&A_dev, a_elems * sizeof(ADataType))) != hipSuccess)
    {
        cleanup();
        return fail_hip("hipMalloc(A)", err);
    }
    if((err = hipMalloc(&B_dev, b_elems * sizeof(BDataType))) != hipSuccess)
    {
        cleanup();
        return fail_hip("hipMalloc(B)", err);
    }
    if((err = hipMalloc(&C_dev, c_elems * sizeof(CDataType))) != hipSuccess)
    {
        cleanup();
        return fail_hip("hipMalloc(C)", err);
    }

    if((err = hipMemcpy(A_dev, A_host, a_elems * sizeof(ADataType), hipMemcpyHostToDevice)) !=
       hipSuccess)
    {
        cleanup();
        return fail_hip("hipMemcpy(A H2D)", err);
    }
    if((err = hipMemcpy(B_dev, B_host, b_elems * sizeof(BDataType), hipMemcpyHostToDevice)) !=
       hipSuccess)
    {
        cleanup();
        return fail_hip("hipMemcpy(B H2D)", err);
    }
    if((err = hipMemset(C_dev, 0, c_elems * sizeof(CDataType))) != hipSuccess)
    {
        cleanup();
        return fail_hip("hipMemset(C)", err);
    }

    float exec_time = -1.0f;
    try
    {
        // k_batch (split-K) is a real capability: Old-TE batched_gemm passes
        // split_k_ through BatchedGemmHostArgs, so we thread it here too
        // (default 1 == no split).
        ck_tile::BatchedGemmHostArgs args{A_dev,
                                          B_dev,
                                          C_dev,
                                          kbatch,
                                          static_cast<ck_tile::index_t>(M),
                                          static_cast<ck_tile::index_t>(N),
                                          static_cast<ck_tile::index_t>(K),
                                          static_cast<ck_tile::index_t>(sa),
                                          static_cast<ck_tile::index_t>(sb),
                                          static_cast<ck_tile::index_t>(sc),
                                          static_cast<ck_tile::index_t>(bsa),
                                          static_cast<ck_tile::index_t>(bsb),
                                          static_cast<ck_tile::index_t>(bsc),
                                          static_cast<ck_tile::index_t>(batch_count)};

        // Build the stream_config to MATCH the Tile Engine batched_gemm profiler
        // (batched_gemm_profiler.hpp: {nullptr, true, log, n_warmup, n_repeat,
        // is_gpu_timer, flush_cache, rotating_count}). The struct field order is
        // {stream_id, time_kernel, log_level, cold_niters, nrepeat,
        // is_gpu_timer, flush_cache, rotating_count} (stream_config.hpp). The
        // benchmark defaults are warmup=50 / repeat=100 / flush_cache=true /
        // rotating_count=1000 / gpu_timer=true; callers thread the exact values
        // through so both sides are driven identically. A caller passing <= 0
        // for warmup/repeat falls back to the TE benchmark defaults.
        const int n_warmup   = warmup > 0 ? static_cast<int>(warmup) : 50;
        const int n_repeat   = repeat > 0 ? static_cast<int>(repeat) : 100;
        const bool do_flush  = flush_cache != 0;
        const int n_rotating = rotating_count > 0 ? static_cast<int>(rotating_count) : 1;
        const ck_tile::stream_config stream{nullptr,
                                            /*time_kernel=*/true,
                                            /*log_level=*/0,
                                            /*cold_niters=*/n_warmup,
                                            /*nrepeat=*/n_repeat,
                                            /*is_gpu_timer=*/true,
                                            /*flush_cache=*/do_flush,
                                            /*rotating_count=*/n_rotating};
        exec_time = SelectedKernel::launch(args, stream);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_batched: kernel launch failed: " << e.what() << std::endl;
        cleanup();
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2;
    }
    catch(...)
    {
        std::cerr << "dispatcher_run_batched: kernel launch failed (unknown exception)"
                  << std::endl;
        cleanup();
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2;
    }

    if((err = hipMemcpy(C_host, C_dev, c_elems * sizeof(CDataType), hipMemcpyDeviceToHost)) !=
       hipSuccess)
    {
        cleanup();
        return fail_hip("hipMemcpy(C D2H)", err);
    }

    if(time_ms)
    {
        *time_ms = exec_time;
    }

    cleanup();
    return 0;
}

void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
