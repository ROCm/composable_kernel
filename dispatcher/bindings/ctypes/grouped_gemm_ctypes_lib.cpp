// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Grouped GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration for the GROUPED GEMM variant.
 * Kernel header included via -include at compile time.
 *
 * The grouped kernel has a genuinely different ABI from regular GEMM: it takes a
 * LIST of (M,N,K) sub-problems plus arrays of A/B/C device pointers, and its
 * generated launch() builds the per-group arg workspace internally:
 *
 *   static float launch(const std::vector<ck_tile::GroupedGemmHostArgs<>>& descs,
 *                       const stream_config& stream);
 *
 * The single-problem dispatcher run path (g_dispatcher->run / GemmHostArgs) cannot
 * express this, and the generated_tile_backend wrapper hard-codes the single-problem
 * launch signature, so this lib calls SelectedKernel::launch(descs, stream) directly
 * and reports the kernel name from the compile-time KERNEL_NAME macro instead of the
 * registry.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_grouped_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_grouped_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

// Kernel header included via -include compiler flag (with CK_TILE_SINGLE_KERNEL_INCLUDE).
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME
// and transitively brings in ck_tile::GroupedGemmHostArgs and ck_tile::stream_config.

// GPU architecture - can be overridden via -DGFX_ARCH="gfx90a" at compile time
#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

static bool g_initialized = false;

// Read an integer benchmark knob from the environment, falling back to
// `fallback` when unset or unparseable. Mirrors generated_tile_backend.hpp so
// both bridge sides honor the same CK_TILE_BENCH_* env vars.
static int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if(v == nullptr || *v == '\0')
        return fallback;
    char* end      = nullptr;
    const long out = std::strtol(v, &end, 10);
    if(end == v)
        return fallback;
    return static_cast<int>(out);
}

extern "C" {

/**
 * Initialize the grouped GEMM library.
 *
 * The grouped path does not use the dispatcher/registry (it launches the
 * force-included kernel directly), so this is a lightweight no-op kept for ABI
 * parity with the regular GEMM lib. Returns 0 on success.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

/**
 * Initialize dispatcher (alias)
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Run grouped GEMM on GPU by launching the force-included kernel directly.
 *
 * For each group: hipMalloc A/B/C, copy A and B host->device, memset C, then build
 * a std::vector<ck_tile::GroupedGemmHostArgs<>> with strides derived from the
 * compile-time ALayout/BLayout/CLayout of the -include'd kernel header (k_batch=1)
 * and launch. After the launch the per-group C buffers are copied back to the
 * caller's host buffers.
 *
 * Layout contract: A is MxK, B is KxN, C is MxN; leading dimensions follow each
 * operand's row/col-major layout (CLayout is always RowMajor for grouped).
 *
 * Returns: 0 on success, -1 on HIP error / generic throw, -2 if the kernel reports
 * the arguments are unsupported.
 */
int dispatcher_run_grouped_gemm(int group_count,
                                const int64_t* Ms,
                                const int64_t* Ns,
                                const int64_t* Ks,
                                const void** A_ptrs,
                                const void** B_ptrs,
                                void** C_ptrs,
                                float* time_ms)
{
    if(!g_initialized || group_count <= 0 || !Ms || !Ns || !Ks || !A_ptrs || !B_ptrs || !C_ptrs)
    {
        return -1;
    }

    std::vector<ADataType*> A_dev(group_count, nullptr);
    std::vector<BDataType*> B_dev(group_count, nullptr);
    std::vector<CDataType*> C_dev(group_count, nullptr);

    auto cleanup_gpu_mem = [&]() {
        for(int g = 0; g < group_count; ++g)
        {
            if(A_dev[g])
                (void)hipFree(A_dev[g]);
            if(B_dev[g])
                (void)hipFree(B_dev[g]);
            if(C_dev[g])
                (void)hipFree(C_dev[g]);
        }
    };

    std::vector<ck_tile::GroupedGemmHostArgs<>> descs;
    descs.reserve(group_count);

    for(int g = 0; g < group_count; ++g)
    {
        const int64_t M = Ms[g];
        const int64_t N = Ns[g];
        const int64_t K = Ks[g];

        if(M <= 0 || N <= 0 || K <= 0 || !A_ptrs[g] || !B_ptrs[g] || !C_ptrs[g])
        {
            cleanup_gpu_mem();
            return -1;
        }

        if(hipMalloc(&A_dev[g], M * K * sizeof(ADataType)) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
        if(hipMalloc(&B_dev[g], K * N * sizeof(BDataType)) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
        if(hipMalloc(&C_dev[g], M * N * sizeof(CDataType)) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }

        if(hipMemcpy(A_dev[g], A_ptrs[g], M * K * sizeof(ADataType), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
        if(hipMemcpy(B_dev[g], B_ptrs[g], K * N * sizeof(BDataType), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
        if(hipMemset(C_dev[g], 0, M * N * sizeof(CDataType)) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }

        // Derive leading dimensions from the compile-time layouts the kernel was
        // generated with (ALayout/BLayout/CLayout from the -include'd header),
        // matching Old-TE gemm_validation_utils.get_abc_layouts:
        //   stride_A = ALayout row-major ? K : M
        //   stride_B = BLayout row-major ? N : K
        //   stride_E = CLayout row-major ? N : M  (CLayout is always RowMajor for grouped)
        using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
        const auto stride_A = std::is_same_v<ALayout, RowMajor> ? static_cast<ck_tile::index_t>(K)
                                                                : static_cast<ck_tile::index_t>(M);
        const auto stride_B = std::is_same_v<BLayout, RowMajor> ? static_cast<ck_tile::index_t>(N)
                                                                : static_cast<ck_tile::index_t>(K);
        const auto stride_E = std::is_same_v<CLayout, RowMajor> ? static_cast<ck_tile::index_t>(N)
                                                                : static_cast<ck_tile::index_t>(M);
        // k_batch=1 for numeric parity.
        descs.emplace_back(static_cast<const void*>(A_dev[g]),
                           static_cast<const void*>(B_dev[g]),
                           std::array<const void*, 0>{},
                           static_cast<void*>(C_dev[g]),
                           /*k_batch=*/1,
                           static_cast<ck_tile::index_t>(M),
                           static_cast<ck_tile::index_t>(N),
                           static_cast<ck_tile::index_t>(K),
                           stride_A,
                           stride_B,
                           std::array<ck_tile::index_t, 0>{},
                           stride_E);
    }

    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_      = nullptr;
    stream_cfg.time_kernel_    = true;
    stream_cfg.log_level_      = 0;
    stream_cfg.cold_niters_    = env_int("CK_TILE_BENCH_WARMUP", 50);
    stream_cfg.nrepeat_        = env_int("CK_TILE_BENCH_REPEAT", 100);
    stream_cfg.is_gpu_timer_   = true;
    stream_cfg.flush_cache_    = false;
    stream_cfg.rotating_count_ = 1;

    float exec_time = 0.0f;
    try
    {
        exec_time = SelectedKernel::launch(descs, stream_cfg);
    }
    catch(const std::exception& e)
    {
        cleanup_gpu_mem();
        if(std::string(e.what()).find("not supported") != std::string::npos)
        {
            if(time_ms)
            {
                *time_ms = -1.0f;
            }
            return -2; // Arguments not supported by this kernel
        }
        return -1;
    }

    // Copy each group's result back to host.
    for(int g = 0; g < group_count; ++g)
    {
        const int64_t M = Ms[g];
        const int64_t N = Ns[g];
        if(hipMemcpy(C_ptrs[g], C_dev[g], M * N * sizeof(CDataType), hipMemcpyDeviceToHost) !=
           hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }

    if(time_ms)
    {
        *time_ms = exec_time;
    }

    cleanup_gpu_mem();
    return 0;
}

/**
 * Get kernel information (legacy single-kernel ABI).
 *
 * Returns the compile-time KERNEL_NAME of the force-included kernel header.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Get the name of the kernel at a given registry index (multi-kernel ABI).
 *
 * Each grouped .so force-includes exactly one kernel header, so index 0 reports
 * KERNEL_NAME and any other index is out of range. Mirrors the regular GEMM lib's
 * name ABI so the Python bridge can use the same name-lookup path.
 * Returns 0 on success, -1 on bad args or out-of-range index.
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

/**
 * Get the number of kernels in this .so (always 1 for the grouped single-include lib).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Cleanup library resources (no-op; kept for ABI parity).
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
