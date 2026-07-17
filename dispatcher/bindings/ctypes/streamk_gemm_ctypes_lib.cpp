// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Stream-K GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration for the STREAM-K GEMM variant.
 * Kernel header included via -include at compile time.
 *
 * Stream-K is a single GEMM (one A/B/C, one M/N/K) like regular GEMM, so this
 * lib keeps the exact same C ABI as gemm_ctypes_lib.cpp -- ``dispatcher_run_gemm``
 * takes host A/B/C and M/N/K. The difference is internal: the generated launch
 * has a Stream-K-specific signature
 *
 *   static float launch(const ck_tile::StreamKHostArgs& args, const stream_config& stream);
 *
 * which allocates the reduction workspace internally (DeviceMem) and uses the
 * Atomic reduction strategy. The single-problem registry path
 * (g_dispatcher->run / GemmHostArgs) and the generated_tile_backend wrapper both
 * hard-code the plain GemmHostArgs launch, so this lib bypasses the registry and
 * calls SelectedKernel::launch(args, stream) directly, reporting the kernel name
 * from the compile-time KERNEL_NAME macro.
 *
 * Because the C ABI matches the regular lib, the Python side reuses
 * GemmDispatcherLib / GpuGemmRunner unchanged -- only the .so internals differ.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_streamk_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>

// Kernel header included via -include compiler flag (with CK_TILE_SINGLE_KERNEL_INCLUDE).
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME
// and transitively brings in ck_tile::StreamKHostArgs and ck_tile::stream_config.

// GPU architecture - can be overridden via -DGFX_ARCH="gfx90a" at compile time
#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

static bool g_initialized = false;

// Read an integer benchmark knob from the environment, falling back to
// `fallback` when unset or unparseable.
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

// Read a boolean benchmark knob ("0"/"false"/"off", any case => false, else true).
static bool env_bool(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    if(v == nullptr || *v == '\0')
        return fallback;
    std::string s(v);
    for(char& c : s)
        if(c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return !(s == "0" || s == "false" || s == "off");
}

extern "C" {

/**
 * Initialize the stream-k GEMM library.
 *
 * The stream-k path does not use the dispatcher/registry (it launches the
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
 * Run a Stream-K GEMM on GPU by launching the force-included kernel directly.
 *
 * hipMalloc A/B/C, copy A and B host->device, memset C (the Atomic reduction
 * strategy accumulates into C, so it must start zeroed), build a
 * ck_tile::StreamKHostArgs whose strides are derived from the kernel's actual
 * ALayout/BLayout/CLayout (no layout hardcoding) and launch. The launch
 * allocates the reduction workspace internally and resets C between timed
 * iterations. C is then copied back.
 *
 * The host buffers must be laid out to match each operand's layout (the Python
 * runner arranges A/B/C as RowMajor=C-contiguous, ColumnMajor=F-contiguous).
 *
 * Returns: 0 on success, -1 on HIP error / generic throw, -2 if the kernel
 * reports the arguments are unsupported.
 */
int dispatcher_run_gemm(
    const void* A, const void* B, void* C, int64_t M, int64_t N, int64_t K, float* time_ms)
{
    if(!g_initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
    {
        return -1;
    }

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    auto cleanup_gpu_mem = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, M * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&B_dev, K * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&C_dev, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Strides are DERIVED from the kernel's actual layouts (ALayout/BLayout/CLayout
    // come from the force-included generated header) -- nothing layout-specific is
    // hardcoded, so every layout (rcr/rrr/ccr/crr/...) works. A RowMajor R x C
    // matrix has leading dim C; a ColumnMajor one has leading dim R.
    //   A is M x K, B is K x N, C is M x N.
    using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
    const ck_tile::index_t lda =
        static_cast<ck_tile::index_t>(std::is_same_v<ALayout, RowMajor> ? K : M);
    const ck_tile::index_t ldb =
        static_cast<ck_tile::index_t>(std::is_same_v<BLayout, RowMajor> ? N : K);
    const ck_tile::index_t ldc =
        static_cast<ck_tile::index_t>(std::is_same_v<CLayout, RowMajor> ? N : M);
    // k_batch is fixed to 1 inside StreamKHostArgs.
    ck_tile::StreamKHostArgs args(static_cast<const void*>(A_dev),
                                  static_cast<const void*>(B_dev),
                                  static_cast<void*>(C_dev),
                                  static_cast<ck_tile::index_t>(M),
                                  static_cast<ck_tile::index_t>(N),
                                  static_cast<ck_tile::index_t>(K),
                                  /*stride_A=*/lda,
                                  /*stride_B=*/ldb,
                                  /*stride_C=*/ldc);

    // Benchmark parameters. warmup/repeat default to old Tile Engine's values
    // (warmup=50, repeat=100); a generous warmup keeps the GPU clock ramped, and
    // 100 timed iterations give a stable median. These were the knobs behind the
    // regular bridge's spurious "perf gap" (#8123): the old default of warmup=3/
    // repeat=10 measured a cold, un-ramped clock. Each knob is env-overridable so
    // a caller can match another harness without recompiling.
    //
    // Divergence from the regular path (generated_tile_backend.hpp): flush_cache_
    // and rotating_count_ default OFF here. The Stream-K Atomic reduction
    // accumulates into C, and the generated launch's launch_kernel_time_mask
    // preprocess re-zeros only the original args.e_ptr -- rotating C across
    // multiple buffers would leave the rotated copies un-zeroed and corrupt the
    // accumulation. Leave rotating_count_=1 unless a caller knows the kernel
    // re-zeros every rotated buffer.
    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_      = nullptr;
    stream_cfg.time_kernel_    = true;
    stream_cfg.log_level_      = 0;
    stream_cfg.cold_niters_    = env_int("CK_TILE_BENCH_WARMUP", 50);
    stream_cfg.nrepeat_        = env_int("CK_TILE_BENCH_REPEAT", 100);
    stream_cfg.is_gpu_timer_   = true;
    stream_cfg.flush_cache_    = env_bool("CK_TILE_BENCH_FLUSH", false);
    stream_cfg.rotating_count_ = env_int("CK_TILE_BENCH_ROTATING", 1);

    float exec_time = 0.0f;
    try
    {
        exec_time = SelectedKernel::launch(args, stream_cfg);
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

    if(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
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
 * Each stream-k .so force-includes exactly one kernel header, so index 0 reports
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
 * Get the number of kernels in this .so (always 1 for the stream-k single-include lib).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Cleanup library resources (no-op; kept for ABI parity).
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
