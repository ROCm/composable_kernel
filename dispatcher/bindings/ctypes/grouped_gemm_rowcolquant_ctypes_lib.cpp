// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm RowColQuant ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE
 * grouped_gemm_rowcolquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *
 * Design: direct launch -- SelectedKernel::launch(vector<QuantGroupedGemmHostArgs>, stream_config,
 * kargs_ptr) is called directly. No dispatcher registry is used: RowColQuant kernels take
 * QuantGroupedGemmHostArgs, which is incompatible with the GeneratedTileKernelInstance::run()
 * signature used by the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 * Each call launches a single problem (num_groups=1). The "grouped" in the name refers
 * to the QuantGroupedGemmHostArgs kernel contract, not multi-group batching by this ABI.
 */

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType,
//          SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// HIP_CHECK calls cleanup() which must be a lambda in scope at every call site.
// All uses of this macro are inside dispatcher_run_gemm, after the
// lambda is defined.
#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            cleanup();                                                                         \
            return -1;                                                                         \
        }                                                                                      \
    }

// GFX_ARCH is normally injected by CMake (-DGFX_ARCH="gfx942"). Define a sentinel if
// it is missing so a hand-rolled compile fails the is_supported_arch() check below with
// a readable message instead of dying on an undeclared identifier.
#ifndef GFX_ARCH
#define GFX_ARCH "unknown"
#endif

// g_ref_count is process-global but scoped to this .so image: each kernel variant
// is compiled into its own .so, so there is no cross-kernel symbol aliasing.
static std::atomic<int> g_ref_count{0};

// Architectures this bridge is known to work on. These fp8/bf8 CompV3 kernels need
// native FP8, so gfx90a is deliberately absent -- it compiles but produces NaN.
// Enabling a new target is a one-line addition here plus a CMake arch entry.
static constexpr const char* kSupportedArchs[] = {"gfx942", "gfx950"};

// True if `arch` starts with any entry of kSupportedArchs. Prefix-matching, because
// hipDeviceProp_t::gcnArchName carries feature suffixes (e.g. "gfx942:sramecc+:xnack-").
static bool is_supported_arch(const std::string& arch)
{
    for(const char* supported : kSupportedArchs)
    {
        if(arch.rfind(supported, 0) == 0)
            return true;
    }
    return false;
}

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_gemm.
 * Returns 0 on success.
 */
int dispatcher_initialize()
{
    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << "dispatcher_initialize: could not query device architecture\n";
        return -1;
    }
    // GFX_ARCH is injected at compile time by CMake (e.g. "gfx942" or "gfx950").
    const std::string arch(props.gcnArchName);
    const std::string compiled_arch(GFX_ARCH);

    // Two distinct checks. First: is the arch this .so was built for one we support at
    // all? A typo or a newly added CMake target would otherwise only surface as a
    // wrong-answer kernel at runtime.
    if(!is_supported_arch(compiled_arch))
    {
        std::cerr << "dispatcher_initialize: compile-time GFX_ARCH '" << compiled_arch
                  << "' is not a supported architecture (supported:";
        for(const char* supported : kSupportedArchs)
            std::cerr << " " << supported;
        std::cerr << ")\n";
        return -1;
    }

    // Second: does the device we are actually running on match that arch? A single-arch
    // .so launched on a different device yields a no-kernel-image failure.
    if(arch.rfind(compiled_arch, 0) != 0)
    {
        std::cerr << "dispatcher_initialize: runtime device architecture '" << arch
                  << "' does not match compile-time GFX_ARCH '" << compiled_arch
                  << "'; this .so was compiled for a different device\n";
        return -1;
    }
    // Increment the reference count. Use fetch_add with release so the
    // device-property checks above are visible to any thread that later
    // reads g_ref_count with acquire ordering.
    g_ref_count.fetch_add(1, std::memory_order_release);
    return 0;
}

/**
 * Short-name alias for dispatcher_initialize(). Every other ctypes lib in this
 * directory exports both spellings; generic loaders (e.g. ctypes_utils.py) bind
 * `dispatcher_init`, so omitting it would make this .so unusable through them.
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Run RowColQuant Grouped GEMM: C[M,N] = dequant(A[M,K], AQ[M,1]) @ dequant(B[K,N], BQ[1,N])
 *
 * A, B, AQ, BQ, C are host pointers to flat packed arrays.
 * For a single-problem (num_groups=1) call, this is equivalent to a standard GEMM with
 * per-row A scales and per-column B scales.
 *
 * Parameters:
 *   A, B, AQ, BQ, C  - host data pointers (flat packed arrays)
 *   M, N, K          - matrix dimensions (single problem)
 *   stride_A         - leading dimension of A (row-major: K)
 *   stride_B         - leading dimension of B (col-major: K)
 *   stride_AQ        - must be 1. Present for ABI symmetry with the other quant ops;
 *                      the RowColQuant kernel hardwires the scale strides and never
 *                      reads this field. Values other than 1 are rejected rather than
 *                      silently ignored.
 *   stride_BQ        - must be 1; see stride_AQ.
 *   stride_C         - leading dimension of C (row-major: N)
 *   QK_A             - number of AQ elements (== M); used only for buffer sizing.
 *   QK_B             - number of BQ elements (== N); used only for buffer sizing.
 *   k_batch          - split-K factor (1 = no split)
 *   time_ms          - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_gemm(const void* A,
                        const void* B,
                        const void* AQ,
                        const void* BQ,
                        void* C,
                        int64_t M,
                        int64_t N,
                        int64_t K,
                        int64_t stride_A,
                        int64_t stride_B,
                        int64_t stride_AQ,
                        int64_t stride_BQ,
                        int64_t stride_C,
                        int64_t QK_A,
                        int64_t QK_B,
                        int k_batch,
                        float* time_ms)
{
    // acquire: synchronise with the release fetch_add in dispatcher_initialize so
    // that all device-property checks performed there are visible here.
    if(g_ref_count.load(std::memory_order_acquire) <= 0)
    {
        std::cerr << "dispatcher_run_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_gemm: invalid dimensions\n";
        return -1;
    }
    if(k_batch <= 0)
    {
        std::cerr << "dispatcher_run_gemm: k_batch must be >= 1, got " << k_batch
                  << " (k_batch is used as a divisor in split-K)\n";
        return -1;
    }
    // RowColQuant kernel indexes M AQ values (per-row) and N BQ values (per-col) directly.
    // Smaller counts cause device out-of-bounds reads; QK_A must equal M, QK_B must equal N.
    if(QK_A != M)
    {
        std::cerr << "dispatcher_run_gemm: QK_A must equal M (" << M
                  << ") for RowColQuant; got QK_A=" << QK_A << "\n";
        return -1;
    }
    if(QK_B != N)
    {
        std::cerr << "dispatcher_run_gemm: QK_B must equal N (" << N
                  << ") for RowColQuant; got QK_B=" << QK_B << "\n";
        return -1;
    }

    // Only packed (contiguous) layouts are supported for A, B, C.
    // stride_AQ and stride_BQ are unused here; the kernel uses broadcast strides (0).
    //
    // B layout (rcr = row-major A, column-major B, row-major C):
    //   B is stored column-major (Fortran order), shape [K, N].
    //   The leading dimension of a column-major [K, N] matrix is K (the number of
    //   rows), so stride_B == K for a packed column-major B.  This is NOT the same
    //   as row-major stride which would be N.  A row-major (C-contiguous) B passed
    //   with stride_B=K would cause the kernel to read the wrong elements.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    // Symmetric with the tensorquant bridge. The kernel ignores these fields entirely
    // (QuantType::RowColQuant builds its AQ/BQ views with literal strides), so accepting
    // an arbitrary value would silently do nothing at all.
    if(stride_AQ != 1 || stride_BQ != 1)
    {
        std::cerr << "dispatcher_run_gemm: stride_AQ and stride_BQ must be 1 (the RowColQuant "
                  << "kernel hardwires its scale strides); got stride_AQ=" << stride_AQ
                  << " stride_BQ=" << stride_BQ << "\n";
        return -1;
    }

    // The ABI takes int64_t but ck_tile::QuantGroupedGemmHostArgs stores ck_tile::index_t
    // (int32_t). Without this check a >2^31 dimension would wrap to a negative extent and
    // the kernel would read out of bounds instead of reporting an error.
    {
        constexpr int64_t kIndexMax =
            static_cast<int64_t>(std::numeric_limits<ck_tile::index_t>::max());
        const int64_t to_narrow[] = {M, N, K, stride_A, stride_B, stride_C};
        for(int64_t v : to_narrow)
        {
            if(v > kIndexMax)
            {
                std::cerr << "dispatcher_run_gemm: dimension or stride " << v << " exceeds the "
                          << kIndexMax << " limit of ck_tile::index_t (int32)\n";
                return -1;
            }
        }
        // M * N and M * K are computed in int64 for the byte counts below, but the kernel
        // also derives tile counts from M and N; a product this large will not fit either.
        if(M > kIndexMax / N || M * N > kIndexMax)
        {
            std::cerr << "dispatcher_run_gemm: M*N (" << M << "*" << N
                      << ") exceeds the range of ck_tile::index_t (int32)\n";
            return -1;
        }
    }

    const ADataType* A_host   = static_cast<const ADataType*>(A);
    const BDataType* B_host   = static_cast<const BDataType*>(B);
    const AQDataType* AQ_host = static_cast<const AQDataType*>(AQ);
    const BQDataType* BQ_host = static_cast<const BQDataType*>(BQ);
    CDataType* C_host         = static_cast<CDataType*>(C);

    ADataType* A_dev   = nullptr;
    BDataType* B_dev   = nullptr;
    AQDataType* AQ_dev = nullptr;
    BQDataType* BQ_dev = nullptr;
    CDataType* C_dev   = nullptr;
    void* kargs_dev    = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(AQ_dev)
            (void)hipFree(AQ_dev);
        if(BQ_dev)
            (void)hipFree(BQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
        if(kargs_dev)
            (void)hipFree(kargs_dev);
    };

    // Allocate device buffers.
    HIP_CHECK(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)));
    // AQ: per-row scale [M, 1] -- QK_A rows, 1 col
    HIP_CHECK(hipMalloc(&AQ_dev, elements_to_bytes<AQDataType>(QK_A)));
    // BQ: per-col scale [1, N] -- 1 row, QK_B cols
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<BQDataType>(QK_B)));
    const std::size_t c_bytes = elements_to_bytes<CDataType>(M * N);
    HIP_CHECK(hipMalloc(&C_dev, c_bytes));

    // Allocate kargs device buffer for grouped GEMM kernel args (1 group)
    HIP_CHECK(hipMalloc(&kargs_dev, sizeof(ck_tile::QuantGemmTransKernelArg)));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(AQ_dev, AQ_host, elements_to_bytes<AQDataType>(QK_A), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(BQ_dev, BQ_host, elements_to_bytes<BQDataType>(QK_B), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, c_bytes));

    // Build QuantGroupedGemmHostArgs for single-group launch.
    //
    // Load-bearing invariant, split across host and kernel:
    //   host  - QK_A == M and QK_B == N were validated at the top of this function, and
    //           the AQ/BQ device buffers were sized from them (M and N elements).
    //   kernel- reads M per-row A scales and N per-col B scales, deriving the counts
    //           from M/N directly.
    //
    // Under QuantType::RowColQuant the kernel builds its scale views with literal
    // strides -- AQ as (M, N) strided (1, 0) and BQ as (M, N) strided (0, 1), see
    // gemm_quant_kernel.hpp -- so the QK_A/QK_B and stride_AQ/stride_BQ fields of the
    // host args are never read. The values below are placeholders; naming them records
    // that they are inert rather than tuned. If a future kernel revision starts
    // honouring them, the names are the thing to grep for.
    constexpr auto kQuantGroupsIgnoredByKernel = static_cast<ck_tile::index_t>(1);
    constexpr auto kScaleStrideIgnoredByKernel = static_cast<ck_tile::index_t>(0);

    ck_tile::QuantGroupedGemmHostArgs args(A_dev,
                                           B_dev,
                                           C_dev,
                                           AQ_dev,
                                           BQ_dev,
                                           static_cast<ck_tile::index_t>(k_batch),
                                           static_cast<ck_tile::index_t>(M),
                                           static_cast<ck_tile::index_t>(N),
                                           static_cast<ck_tile::index_t>(K),
                                           kQuantGroupsIgnoredByKernel, // QK_A
                                           kQuantGroupsIgnoredByKernel, // QK_B
                                           static_cast<ck_tile::index_t>(stride_A),
                                           static_cast<ck_tile::index_t>(stride_B),
                                           static_cast<ck_tile::index_t>(stride_C),
                                           kScaleStrideIgnoredByKernel,  // stride_AQ
                                           kScaleStrideIgnoredByKernel); // stride_BQ

    const std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs = {args};

    const bool do_time = (time_ms != nullptr);
    // stream_config fields, in declaration order (see ck_tile/host/stream_config.hpp):
    //   stream_id_, time_kernel_, log_level_, cold_niters_, nrepeat_,
    //   is_gpu_timer_, flush_cache_, rotating_count_
    // Note there is no do_log_perf member; is_gpu_timer_ selects hipEvent timing over
    // wall-clock, and flush_cache_ enables the rotating-buffer cache flush.
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

    // Split-K selects the atomic_add epilogue, so C must start at zero before *every*
    // launch -- not just the first. With timing enabled the kernel runs
    // cold_niters_ + nrepeat_ times, and a C zeroed only once would come back holding
    // the sum of all of them. SelectedKernel::launch forwards this hook to
    // ck_tile::launch_kernel_time_mask, which calls it before each invocation.
    // For k_batch == 1 the epilogue is `set` and repeated launches are idempotent, so
    // the memset is skipped to keep the timing loop measuring only the kernel.
    hipError_t clear_err = hipSuccess;
    auto clear_c         = [&]() {
        if(k_batch > 1)
        {
            hipError_t e = hipMemsetAsync(C_dev, 0, c_bytes, stream_cfg.stream_id_);
            // Record the first failure rather than aborting: this runs inside the
            // kernel-launch helper, which has no way to propagate an error out.
            if(e != hipSuccess && clear_err == hipSuccess)
                clear_err = e;
        }
    };

    float exec_time = -1.0f;
    try
    {
        exec_time = SelectedKernel::launch(gemm_descs, stream_cfg, kargs_dev, clear_c);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_gemm: kernel launch threw: " << e.what() << "\n";
        cleanup();
        return -3;
    }
    catch(...)
    {
        std::cerr << "dispatcher_run_gemm: kernel launch threw unknown exception\n";
        cleanup();
        return -3;
    }

    if(clear_err != hipSuccess)
    {
        std::cerr << "dispatcher_run_gemm: failed to clear C between split-K launches: "
                  << hipGetErrorString(clear_err) << "\n";
        cleanup();
        return -1;
    }

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    HIP_CHECK(hipMemcpy(C_host, C_dev, c_bytes, hipMemcpyDeviceToHost));

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

/**
 * Return the compile-time KERNEL_NAME of the force-included kernel.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

// This bridge is one-.so-per-kernel by construction: the build force-includes exactly
// one generated header via `hipcc -include <kernel.hpp>`, giving one SelectedKernel.
// Scaling to N kernels means N .so files (the pattern bquant/aquant/abquant follow),
// not incrementing this constant. A generated header may override it via -D if needed.
#ifndef CK_TILE_DISPATCHER_KERNEL_COUNT
#define CK_TILE_DISPATCHER_KERNEL_COUNT 1
#endif

/**
 * Number of kernels compiled into this .so.
 */
int dispatcher_get_kernel_count() { return CK_TILE_DISPATCHER_KERNEL_COUNT; }

/**
 * Decrement the initialisation reference count. When it reaches zero the library
 * is considered uninitialised and the next call to dispatcher_run_gemm
 * will fail until dispatcher_initialize() is called again.
 *
 * Using a reference count instead of a boolean allows multiple independent Python
 * wrappers to share the same loaded .so without one wrapper's destructor
 * invalidating another live wrapper.
 *
 * This function does not free any GPU memory or unload the library; those are
 * managed per-call inside dispatcher_run_gemm.
 */
void dispatcher_cleanup()
{
    // Only decrement if already positive to guard against unpaired cleanup calls.
    int prev = g_ref_count.load(std::memory_order_relaxed);
    while(prev > 0 && !g_ref_count.compare_exchange_weak(
                          prev, prev - 1, std::memory_order_release, std::memory_order_relaxed))
        ; // retry on CAS failure
}

} // extern "C"
