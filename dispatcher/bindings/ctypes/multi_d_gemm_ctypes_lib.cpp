// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Multi-D GEMM Dispatcher ctypes Library
 *
 * Provides a C API for Python ctypes integration for the MULTI_D GEMM variant.
 * Kernel header included via -include at compile time.
 *
 * The multi-D kernel has a genuinely different ABI from regular GEMM: in
 * addition to A/B/C it consumes a fixed number (NumDTensor) of extra D device
 * pointers that the CShuffle epilogue fuses element-wise into the output:
 *
 *   E = elementwise_op(A @ B, D0, D1, ...)
 *
 * Its generated launch() takes GemmMultiDArgs (= GemmMultiDHostArgs<NumDTensor>)
 * carrying the D-pointer array and per-D strides:
 *
 *   static float launch(const GemmMultiDArgs& args, const stream_config& stream);
 *
 * The single-problem dispatcher run path (g_dispatcher->run / GemmHostArgs)
 * cannot express the D tensors, and the generated_tile_backend wrapper ignores
 * d_ptrs and calls the GemmHostArgs overload (empty D tensors), so this lib
 * calls SelectedKernel::launch(GemmMultiDArgs, stream) directly and reports the
 * kernel name from the compile-time KERNEL_NAME macro instead of the registry.
 * This mirrors the grouped / stream-K bridge libraries.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_multi_d_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_multi_d_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

// Kernel header included via -include compiler flag (with CK_TILE_SINGLE_KERNEL_INCLUDE).
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME,
// NumDTensor, DsDataType, DsLayout, DLayout, ElementWiseFn, GemmMultiDArgs, and
// transitively brings in ck_tile::GemmMultiDHostArgs and ck_tile::stream_config.

// GPU architecture - must be provided via -DGFX_ARCH="<arch>" at compile time
#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

#ifndef GEMM_KEY_NUM_D_TENSORS
#define GEMM_KEY_NUM_D_TENSORS 0
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

// Read a boolean benchmark knob from the environment. Accepts 1/0, true/false,
// yes/no, on/off (case-insensitive). Falls back to `fallback` when unset.
static bool env_bool(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    if(v == nullptr || *v == '\0')
        return fallback;
    std::string s(v);
    for(auto& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(s == "1" || s == "true" || s == "yes" || s == "on")
        return true;
    if(s == "0" || s == "false" || s == "no" || s == "off")
        return false;
    return fallback;
}

extern "C" {

/**
 * Initialize the multi-D GEMM library.
 *
 * The multi-D path does not use the dispatcher/registry (it launches the
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
 * Number of D tensors this kernel was compiled for.
 *
 * The Python runner queries this so it can allocate/pass exactly the right
 * number of D operands (the count is baked into the force-included header).
 */
int dispatcher_get_num_d_tensors() { return static_cast<int>(NumDTensor); }

/**
 * Run multi-D GEMM on GPU by launching the force-included kernel directly.
 *
 * hipMalloc A/B/C plus `num_d` D buffers, copy A/B and each D host->device,
 * memset C, build a GemmMultiDArgs with strides derived from the compile-time
 * ALayout/BLayout/CLayout/DLayout of the -include'd header (k_batch=1), launch,
 * then copy C back.
 *
 * Layout contract: A is MxK, B is KxN, C and every D are MxN; leading
 * dimensions follow each operand's row/col-major layout. C and D are row-major
 * for the TE multi_d builder (4-char layout, last two chars 'r').
 *
 * `d_ptrs` points to `num_d` host buffers (each MxN, element type == CDataType).
 * `num_d` MUST equal dispatcher_get_num_d_tensors(); a mismatch returns -1.
 *
 * Returns: 0 on success, -1 on HIP error / bad args / generic throw, -2 if the
 * kernel reports the arguments are unsupported.
 */
int dispatcher_run_multi_d_gemm(const void* A,
                                const void* B,
                                const void** d_ptrs,
                                int num_d,
                                void* C,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                float* time_ms)
{
    if(!g_initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
    {
        return -1;
    }
    if(num_d != static_cast<int>(NumDTensor))
    {
        return -1; // caller must pass exactly the compiled-in number of D tensors
    }
    if(NumDTensor > 0 && !d_ptrs)
    {
        return -1;
    }

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;
    std::array<CDataType*, NumDTensor> D_dev{};
    for(std::size_t i = 0; i < NumDTensor; ++i)
        D_dev[i] = nullptr;

    auto cleanup_gpu_mem = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
        for(std::size_t i = 0; i < NumDTensor; ++i)
            if(D_dev[i])
                (void)hipFree(D_dev[i]);
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
    for(std::size_t i = 0; i < NumDTensor; ++i)
    {
        if(!d_ptrs[i])
        {
            cleanup_gpu_mem();
            return -1;
        }
        if(hipMalloc(&D_dev[i], M * N * sizeof(CDataType)) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }

    if(hipMemcpy(A_dev, A, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemcpy(B_dev, B, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    for(std::size_t i = 0; i < NumDTensor; ++i)
    {
        if(hipMemcpy(D_dev[i], d_ptrs[i], M * N * sizeof(CDataType), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }

    // Derive leading dimensions from the compile-time layouts the kernel was
    // generated with (ALayout/BLayout/CLayout/DLayout from the -include'd
    // header), matching Old-TE gemm_validation_utils.get_abc_layouts:
    //   stride_A = ALayout row-major ? K : M
    //   stride_B = BLayout row-major ? N : K
    //   stride_D = DLayout row-major ? N : M  (row-major for multi_d)
    //   stride_E = CLayout row-major ? N : M  (row-major for multi_d)
    using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
    const auto stride_A = std::is_same_v<ALayout, RowMajor> ? static_cast<ck_tile::index_t>(K)
                                                            : static_cast<ck_tile::index_t>(M);
    const auto stride_B = std::is_same_v<BLayout, RowMajor> ? static_cast<ck_tile::index_t>(N)
                                                            : static_cast<ck_tile::index_t>(K);
    const auto stride_E = std::is_same_v<CLayout, RowMajor> ? static_cast<ck_tile::index_t>(N)
                                                            : static_cast<ck_tile::index_t>(M);
    const auto stride_D = std::is_same_v<DLayout, RowMajor> ? static_cast<ck_tile::index_t>(N)
                                                            : static_cast<ck_tile::index_t>(M);

    std::array<const void*, NumDTensor> ds_ptr{};
    std::array<ck_tile::index_t, NumDTensor> stride_Ds{};
    for(std::size_t i = 0; i < NumDTensor; ++i)
    {
        ds_ptr[i]    = static_cast<const void*>(D_dev[i]);
        stride_Ds[i] = stride_D;
    }

    // k_batch=1 for numeric parity (multi_d kernel requires k_batch == 1).
    GemmMultiDArgs args(static_cast<const void*>(A_dev),
                        static_cast<const void*>(B_dev),
                        ds_ptr,
                        static_cast<void*>(C_dev),
                        /*k_batch=*/1,
                        static_cast<ck_tile::index_t>(M),
                        static_cast<ck_tile::index_t>(N),
                        static_cast<ck_tile::index_t>(K),
                        stride_A,
                        stride_B,
                        stride_Ds,
                        stride_E);

    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_    = nullptr;
    stream_cfg.time_kernel_  = true;
    stream_cfg.log_level_    = 0;
    stream_cfg.cold_niters_  = env_int("CK_TILE_BENCH_WARMUP", 50);
    stream_cfg.nrepeat_      = env_int("CK_TILE_BENCH_REPEAT", 100);
    stream_cfg.is_gpu_timer_ = true;
    // Fair-by-default: match Old-TE gemm_multi_d benchmark (flush_cache=true,
    // rotating_count=1000) so the committed bridge benchmark is reproducible and
    // apples-to-apples out of the box. Both remain env-tunable.
    stream_cfg.flush_cache_    = env_bool("CK_TILE_BENCH_FLUSH", true);
    stream_cfg.rotating_count_ = env_int("CK_TILE_BENCH_ROTATING", 1000);

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

    if(hipMemcpy(C, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
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
 * Each multi-D .so force-includes exactly one kernel header, so index 0 reports
 * KERNEL_NAME and any other index is out of range. Mirrors the regular GEMM
 * lib's name ABI so the Python bridge can use the same name-lookup path.
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
 * Get the number of kernels in this .so (always 1 for the multi-D single-include lib).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Cleanup library resources (no-op; kept for ABI parity).
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
