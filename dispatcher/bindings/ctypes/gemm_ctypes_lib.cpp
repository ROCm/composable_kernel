// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Kernel header included via -include at compile time.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Host-side B-preshuffle utilities. For a weight-preshuffled kernel the device
// expects B already reordered into the pipeline's packed layout; this is the
// SAME transform Old-TE's gemm_preshuffle profiler applies (shuffle_b /
// shuffle_b_permuteN in tensor_shuffle_utils.hpp) so the bridge produces
// byte-for-byte identical B, hence identical results.
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_permute.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"

// Kernel header included via -include compiler flag
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME

// GPU architecture - REQUIRED at compile time (pass -DGFX_ARCH=<arch>).
// The arch is resolved from the host at build time (see gemm_utils.py
// _resolve_arch / rocminfo); it is never silently defaulted to a specific GPU.
#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

#if defined(GEMM_KEY_DTYPE_A)
// Host-side B-preshuffle utilities.
//
// Whether a kernel preshuffles its B operand is a CAPABILITY THE KERNEL CARRIES
// IN ITS OWN METADATA -- the codegen emits `static constexpr bool
// SelectedKernel::Preshuffle` (unified_gemm_codegen.py) on every kernel config.
// The B-upload site below branches on that trait via `if constexpr`, so there is
// no `GEMM_KEY_PRESHUFFLE` capability macro. These helpers are templated on the
// kernel type so they are only instantiated for a preshuffled kernel: the
// non-preshuffle `if constexpr` branch is discarded before instantiation, so
// shuffle_b is never instantiated for a tile geometry that was never meant to be
// shuffled. The one remaining preprocessor guard (`GEMM_KEY_DTYPE_A`) only asks
// "is this a modern codegen header?" -- it mirrors the legacy-header fallback in
// dispatcher_initialize() and is NOT a preshuffle capability switch.

// Adapter exposing the force-included kernel's tile geometry under the field
// names ck_tile::shuffle_b / shuffle_b_permuteN expect. Mirrors Old-TE's
// gemm_preshuffle_benchmark.hpp::KernelConfig so the permutation is identical.
template <typename Kernel>
struct BridgePreshuffleConfig
{
    static constexpr ck_tile::index_t M_Tile = Kernel::TileM;
    static constexpr ck_tile::index_t N_Tile = Kernel::TileN;
    static constexpr ck_tile::index_t K_Tile = Kernel::TileK;

    static constexpr ck_tile::index_t M_Warp = Kernel::WarpPerBlock_M;
    static constexpr ck_tile::index_t N_Warp = Kernel::WarpPerBlock_N;
    static constexpr ck_tile::index_t K_Warp = Kernel::WarpPerBlock_K;

    static constexpr ck_tile::index_t M_Warp_Tile = Kernel::WarpTileM;
    static constexpr ck_tile::index_t N_Warp_Tile = Kernel::WarpTileN;
    static constexpr ck_tile::index_t K_Warp_Tile = Kernel::WarpTileK;

    static constexpr bool permuteN = Kernel::PermuteN;
};

// Preshuffle host B into the packed layout the device pipeline reads. Returns a
// contiguous host buffer of the shuffled bytes.
//
// The shuffle utils (shuffle_b / shuffle_b_permuteN) take a rank-2 HostTensor
// with lengths {K, N} whose PHYSICAL buffer is N-outer / K-contiguous -- exactly
// Old-TE's b_k_n, built as host_tensor_descriptor(K, N, stride, is_row_major=
// false) for the rcr kernel's column-major BLayout. The bridge runner hands B in
// this same order for a 'c' B operand (ascontiguousarray(B.T), shape [N, K] row-
// major == column-major [K, N]), so filling the col-major {K, N} tensor's flat
// storage directly reproduces Old-TE's b_k_n byte-for-byte, hence an identical
// permutation and identical results.
template <typename Kernel, typename T>
static ck_tile::HostTensor<T> preshuffle_host_b(const T* b_host, int64_t K, int64_t N)
{
    using Config = BridgePreshuffleConfig<Kernel>;
    // Build b_k_n with the SAME descriptor Old-TE uses:
    //   host_tensor_descriptor(K, N, stride_b, is_row_major(BLayout))
    // host_tensor_descriptor takes a compile-time bool_constant. BLayout is the
    // force-included kernel's own B layout alias; for the rcr preshuffle kernel
    // it is column-major, giving lengths {K, N} with strides {1, K} (N-outer,
    // K-contiguous) -- the exact physical order shuffle_b / shuffle_b_permuteN
    // expect and that the runner supplies for a 'c' B operand.
    constexpr bool kBRowMajor = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
    // Byte-identity correctness contract: the whole shuffle argument (the runner
    // hands B for a 'c'/column-major operand and we fill the {K,N} tensor's flat
    // storage directly to reproduce Old-TE's b_k_n byte-for-byte) is only valid
    // when BLayout is column-major. If a future layout expansion force-includes a
    // row-major-B kernel here, the copy order would be transposed and B silently
    // mis-shuffled. Fail loudly at compile time instead of producing wrong
    // results. (Preshuffle scope is pinned rcr today, so this always holds.)
    static_assert(!kBRowMajor,
                  "preshuffle_host_b requires a column-major BLayout for byte-identity "
                  "with the preshuffle kernel; a row-major B would be silently "
                  "mis-shuffled. Extend this path before enabling a row-major-B layout.");
    const auto stride_b = ck_tile::get_default_stride(static_cast<ck_tile::index_t>(K),
                                                      static_cast<ck_tile::index_t>(N),
                                                      0,
                                                      ck_tile::bool_constant<kBRowMajor>{});
    ck_tile::HostTensor<T> b_k_n(
        ck_tile::host_tensor_descriptor(static_cast<ck_tile::index_t>(K),
                                        static_cast<ck_tile::index_t>(N),
                                        stride_b,
                                        ck_tile::bool_constant<kBRowMajor>{}));
    // b_host is a raw byte buffer from Python/ctypes (e.g. fp8/bf8 handed as
    // numpy uint8) reinterpreted as T. A typed element copy (std::copy) performs
    // typed reads through that pointer -- strict-aliasing / object-lifetime UB for
    // the raw bytes. memcpy the exact byte span into the tensor's flat storage
    // instead: byte-identical, with no typed reads from the caller's buffer.
    std::memcpy(b_k_n.data(), b_host, static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(T));
    if constexpr(Config::permuteN)
    {
        return ck_tile::shuffle_b_permuteN<Config>(b_k_n);
    }
    else
    {
        return ck_tile::shuffle_b<Config>(b_k_n);
    }
}

// Shuffled-B cache key: the identity of the source B plus the transform that
// produced the shuffled bytes. permute_n distinguishes shuffle_b (false) from
// shuffle_b_permuteN (true), so the two transforms never alias one entry if a
// process ever mixes them.
struct ShuffleKey
{
    const void* ptr;
    int64_t K;
    int64_t N;
    bool permute_n;
    bool operator==(const ShuffleKey& o) const
    {
        return ptr == o.ptr && K == o.K && N == o.N && permute_n == o.permute_n;
    }
};

struct ShuffleKeyHash
{
    size_t operator()(const ShuffleKey& k) const noexcept
    {
        auto mix = [](size_t h, size_t v) {
            return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        };
        size_t h = std::hash<const void*>{}(k.ptr);
        h        = mix(h, std::hash<int64_t>{}(k.K));
        h        = mix(h, std::hash<int64_t>{}(k.N));
        h        = mix(h, std::hash<bool>{}(k.permute_n));
        return h;
    }
};

// Multi-entry shuffled-B cache. SAFE BY DEFAULT: OFF unless explicitly opted
// into, so every dispatcher_run_gemm recomputes the host shuffle from the B the
// caller actually passed. This is the only correct default across the public
// Python API, where GpuGemmRunner.run() builds a fresh (encoded) B temporary on
// every call: once that temporary is freed, numpy may hand back the SAME address
// for a different same-shaped B, so a pointer-keyed entry cannot distinguish them
// and would silently serve stale weights.
//
// OPT-IN CACHE (perf sweeps only): set CK_DISPATCHER_PRESHUFFLE_CACHE=1 to reuse
// the shuffle across calls, keyed on {ptr, K, N, transform}. Unlike a single
// slot -- which a multi-shape A/B sweep evicts on every shape change, dropping
// the hit rate to ~0% -- the map keeps one entry per distinct shape so the cache
// actually pays off. Valid ONLY under a strict IMMUTABILITY CONTRACT: for a fixed
// (b_host, K, N) the bytes behind *b_host must stay immutable and the B object
// kept alive across the repeated calls. The A/B perf sweep honours this (one B
// per shape, never mutated or freed between iterations).
using ShuffledBCache =
    std::unordered_map<ShuffleKey, std::shared_ptr<ck_tile::HostTensor<BDataType>>, ShuffleKeyHash>;
[[maybe_unused]] static ShuffledBCache g_shuffled_b_cache;

// Whether the opt-in cache is enabled. OFF by default; resolved once from the
// environment. Only a caller that guarantees the IMMUTABILITY CONTRACT above
// (perf sweep) should turn this on.
[[maybe_unused]] static bool preshuffle_cache_enabled()
{
    static const bool enabled = []() {
        const char* v = std::getenv("CK_DISPATCHER_PRESHUFFLE_CACHE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// Return a pointer to the shuffled bytes for this B. By default recomputes the
// shuffle every call (safe: never serves stale bytes), returning a thread_local
// scratch buffer that stays alive until the next call. Reuses the map only when
// CK_DISPATCHER_PRESHUFFLE_CACHE is set. Not thread-safe (bridge is single-
// threaded), which matches the rest of this translation unit.
template <typename Kernel>
static const BDataType* get_shuffled_b(const BDataType* b_host, int64_t K, int64_t N)
{
    constexpr bool permute_n = BridgePreshuffleConfig<Kernel>::permuteN;

    if(!preshuffle_cache_enabled())
    {
        static thread_local std::shared_ptr<ck_tile::HostTensor<BDataType>> scratch;
        scratch = std::make_shared<ck_tile::HostTensor<BDataType>>(
            preshuffle_host_b<Kernel, BDataType>(b_host, K, N));
        return scratch->data();
    }

    const ShuffleKey key{b_host, K, N, permute_n};
    auto it = g_shuffled_b_cache.find(key);
    if(it == g_shuffled_b_cache.end())
    {
        it = g_shuffled_b_cache
                 .emplace(key,
                          std::make_shared<ck_tile::HostTensor<BDataType>>(
                              preshuffle_host_b<Kernel, BDataType>(b_host, K, N)))
                 .first;
    }
    return it->second->data();
}
#endif // GEMM_KEY_DTYPE_A

// Global dispatcher (initialized once, managed via shared_ptr for safe cleanup)
static std::shared_ptr<Dispatcher> g_dispatcher = nullptr;
static bool g_initialized                       = false;

#define HIP_CHECK(call)        \
    {                          \
        hipError_t err = call; \
        if(err != hipSuccess)  \
        {                      \
            return -1;         \
        }                      \
    }

extern "C" {

/**
 * Initialize dispatcher with a kernel
 * Must be called before run_gemm
 *
 * Returns: 0 on success, -1 on error
 */
int dispatcher_initialize()
{
    if(g_initialized)
    {
        return 0; // Already initialized
    }

    // Create kernel key from the force-included kernel header.
    //
    // The GEMM_KEY_* macros are emitted by the codegen into the force-included
    // header (see unified_gemm_codegen.py, CK_TILE_SINGLE_KERNEL_INCLUDE block).
    // Building the key from them makes the registry entry truthful: it reflects
    // THIS kernel's real dtypes/layouts/tile/traits instead of a hard-coded
    // fp16/rcr/128x128x32 default. Enum fields use the string_to_* helpers from
    // kernel_key.hpp, whose accepted strings match the codegen's emitted values
    // byte-for-byte.
    KernelKey key;
#ifdef GEMM_KEY_DTYPE_A
    key.signature.dtype_a             = string_to_dtype(GEMM_KEY_DTYPE_A);
    key.signature.dtype_b             = string_to_dtype(GEMM_KEY_DTYPE_B);
    key.signature.dtype_c             = string_to_dtype(GEMM_KEY_DTYPE_C);
    key.signature.dtype_acc           = string_to_dtype(GEMM_KEY_DTYPE_ACC);
    key.signature.layout_a            = string_to_layout(GEMM_KEY_LAYOUT_A);
    key.signature.layout_b            = string_to_layout(GEMM_KEY_LAYOUT_B);
    key.signature.layout_c            = string_to_layout(GEMM_KEY_LAYOUT_C);
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = (GEMM_KEY_GROUPED != 0);
    key.signature.split_k             = GEMM_KEY_SPLIT_K;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {GEMM_KEY_TILE_M, GEMM_KEY_TILE_N, GEMM_KEY_TILE_K};
    key.algorithm.wave_shape      = {GEMM_KEY_WAVE_M, GEMM_KEY_WAVE_N, GEMM_KEY_WAVE_K};
    key.algorithm.warp_tile_shape = {
        GEMM_KEY_WARP_TILE_M, GEMM_KEY_WARP_TILE_N, GEMM_KEY_WARP_TILE_K};
    key.algorithm.pipeline      = string_to_pipeline(GEMM_KEY_PIPELINE);
    key.algorithm.scheduler     = string_to_scheduler(GEMM_KEY_SCHEDULER);
    key.algorithm.epilogue      = string_to_epilogue(GEMM_KEY_EPILOGUE);
    key.algorithm.block_size    = GEMM_KEY_BLOCK_SIZE;
    key.algorithm.double_buffer = (GEMM_KEY_DOUBLE_BUFFER != 0);
    key.algorithm.persistent    = (GEMM_KEY_PERSISTENT != 0);
    // Read the preshuffle capability from the kernel's own metadata trait rather
    // than the GEMM_KEY_PRESHUFFLE macro -- same value (both emitted by codegen),
    // but the capability lives with the kernel, consistent with the B-upload path.
    key.algorithm.preshuffle      = SelectedKernel::Preshuffle;
    key.algorithm.transpose_c     = (GEMM_KEY_TRANSPOSE_C != 0);
    key.algorithm.num_wave_groups = GEMM_KEY_NUM_WAVE_GROUPS;
    // pad_m/n/k participate in both the key's hash/equality and the kernel
    // name, so they must be derived from the codegen macros too -- otherwise a
    // kernel built with padding disabled would register under a key claiming
    // pad=true and disagree with its own name.
    key.algorithm.pad_m = (GEMM_KEY_PAD_M != 0);
    key.algorithm.pad_n = (GEMM_KEY_PAD_N != 0);
    key.algorithm.pad_k = (GEMM_KEY_PAD_K != 0);
    key.gfx_arch        = GFX_ARCH;
#else
    // Fallback default for headers generated before GEMM_KEY_* macros existed
    // (fp16 / rcr / compv4-cshuffle-intrawave, 128x128x32). The macro path
    // above is the source of truth for any freshly generated kernel.
    key.signature.dtype_a   = DataType::FP16;
    key.signature.dtype_b   = DataType::FP16;
    key.signature.dtype_c   = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    // Derive A/B/C layouts from the force-included kernel's own layout types
    // instead of hardcoding rcr. The dispatcher's supports() gate is layout-aware
    // (it only constrains a dimension that an operand's inner axis maps to), so a
    // wrong key layout makes it reject valid problems -- e.g. a crr kernel does not
    // gate K, but with a hardcoded rcr key supports() would apply rcr's K-gate and
    // reject TileK=192 problems that Old-TE runs. ALayout/BLayout/CLayout are the
    // global aliases exported by the kernel header under CK_TILE_SINGLE_KERNEL_INCLUDE.
    using RowMajorLayout = ck_tile::tensor_layout::gemm::RowMajor;
    key.signature.layout_a =
        std::is_same_v<ALayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.layout_b =
        std::is_same_v<BLayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.layout_c =
        std::is_same_v<CLayout, RowMajorLayout> ? LayoutTag::RowMajor : LayoutTag::ColMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {128, 128, 32};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline        = Pipeline::CompV4;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = true;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = false;
    key.algorithm.transpose_c     = false;
    key.algorithm.num_wave_groups = 1;
    key.gfx_arch                  = GFX_ARCH;
#endif // GEMM_KEY_DTYPE_A

    // Register kernel using types from force-included header
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    // Create dispatcher (using shared_ptr for safe memory management)
    g_dispatcher  = std::make_shared<Dispatcher>();
    g_initialized = true;

    return 0;
}

/**
 * Get kernel tile configuration
 */
int dispatcher_get_kernel_config(int* tile_m,
                                 int* tile_n,
                                 int* tile_k,
                                 int* warp_tile_m,
                                 int* warp_tile_n,
                                 int* warp_tile_k,
                                 int* warp_m,
                                 int* warp_n,
                                 int* warp_k)
{
    if(!g_initialized)
    {
        return -1;
    }

    auto kernels = Registry::instance().get_all();
    if(kernels.empty())
    {
        return -1;
    }

    // Get configuration from first kernel
    auto& key  = kernels[0]->get_key();
    auto& algo = key.algorithm;

    if(tile_m)
        *tile_m = algo.tile_shape.m;
    if(tile_n)
        *tile_n = algo.tile_shape.n;
    if(tile_k)
        *tile_k = algo.tile_shape.k;
    if(warp_tile_m)
        *warp_tile_m = algo.warp_tile_shape.m;
    if(warp_tile_n)
        *warp_tile_n = algo.warp_tile_shape.n;
    if(warp_tile_k)
        *warp_tile_k = algo.warp_tile_shape.k;
    if(warp_m)
        *warp_m = algo.wave_shape.m;
    if(warp_n)
        *warp_n = algo.wave_shape.n;
    if(warp_k)
        *warp_k = algo.wave_shape.k;

    return 0;
}

/**
 * Get the selected kernel name for a problem
 */
int dispatcher_select_kernel(int64_t M, int64_t N, int64_t K, char* name_buffer, int buffer_size)
{
    if(!g_initialized || !name_buffer || buffer_size <= 0)
    {
        return -1;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);

    if(!kernel)
    {
        return -1;
    }

    std::string name = kernel->get_name();
    strncpy(name_buffer, name.c_str(), buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';

    return 0;
}

/**
 * Check if a problem size is supported by available kernels
 */
int dispatcher_is_supported(int64_t M, int64_t N, int64_t K)
{
    if(!g_initialized)
    {
        return 0;
    }

    if(M <= 0 || N <= 0 || K <= 0)
    {
        return 0;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    return kernel != nullptr ? 1 : 0;
}

/**
 * Run GEMM on GPU via dispatcher
 *
 * PRESHUFFLE (weight-preshuffled kernels only):
 *   The host B-shuffle is recomputed every call by default, so the kernel always
 *   runs on the B the caller passed -- safe with no lifetime assumptions. Perf
 *   sweeps that reuse one immutable B per shape may set the env var
 *   CK_DISPATCHER_PRESHUFFLE_CACHE=1 to reuse the shuffle across calls (keyed on
 *   (B pointer, K, N)); under that opt-in the caller MUST keep B alive and its
 *   bytes immutable for a fixed (B, K, N), or a reused address silently computes
 *   on stale weights. Non-preshuffle kernels are unaffected.
 */
int dispatcher_run_gemm(
    const void* A, const void* B, void* C, int64_t M, int64_t N, int64_t K, float* time_ms)
{
    if(!g_initialized || !A || !B || !C)
    {
        return -1;
    }

    // Reject non-positive dims before any size arithmetic: a negative M/N/K from a
    // mis-marshaled caller would otherwise wrap to an enormous size_t byte count in
    // the hipMalloc/hipMemcpy calls below. (Matches the guard in
    // dispatcher_is_supported.)
    if(M <= 0 || N <= 0 || K <= 0)
    {
        return -1;
    }

    // First check if any kernel supports this problem
    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    if(!kernel)
    {
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2; // No suitable kernel
    }

    // Cast to correct types (from force-included header)
    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    // Allocate GPU memory
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

    if(hipMalloc(&A_dev, static_cast<size_t>(M) * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&B_dev, static_cast<size_t>(K) * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&C_dev, static_cast<size_t>(M) * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Copy input data to GPU
    if(hipMemcpy(
           A_dev, A_host, static_cast<size_t>(M) * K * sizeof(ADataType), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
#if defined(GEMM_KEY_DTYPE_A)
    // Metadata-driven B upload: the kernel's own Preshuffle trait (emitted by the
    // codegen as SelectedKernel::Preshuffle) decides whether B is reordered on the
    // host before the copy -- no capability macro. The unused branch is discarded
    // at compile time, so the shuffle helpers only instantiate for preshuffle
    // kernels (identical dead-code elimination the old #if GEMM_KEY_PRESHUFFLE
    // gave), but the capability is now read from kernel metadata.
    if constexpr(SelectedKernel::Preshuffle)
    {
        // Weight-preshuffled kernel: reorder B on the host into the packed layout
        // the device pipeline reads, exactly as Old-TE does before launch. The
        // shuffle is a pure permutation (same element count), so the device buffer
        // size is unchanged. B_host stays the logical (unshuffled) B so the
        // Python-side numpy reference (A @ B) remains valid. Recomputed each call
        // by default (safe); reused across calls only when
        // CK_DISPATCHER_PRESHUFFLE_CACHE is set (perf sweep, immutable B). Either
        // way the shuffle runs here, before the timed g_dispatcher->run() below,
        // so it never affects the kernel measurement.
        const BDataType* b_shuffled = get_shuffled_b<SelectedKernel>(B_host, K, N);
        if(hipMemcpy(B_dev,
                     b_shuffled,
                     static_cast<size_t>(K) * N * sizeof(BDataType),
                     hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }
    else
    {
        if(hipMemcpy(B_dev,
                     B_host,
                     static_cast<size_t>(K) * N * sizeof(BDataType),
                     hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup_gpu_mem();
            return -1;
        }
    }
#else
    // Legacy header (pre-GEMM_KEY_* codegen): no Preshuffle trait and no
    // preshuffle kernels are generated for it -- copy B verbatim.
    if(hipMemcpy(
           B_dev, B_host, static_cast<size_t>(K) * N * sizeof(BDataType), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
#endif // GEMM_KEY_DTYPE_A
    if(hipMemset(C_dev, 0, static_cast<size_t>(M) * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Run GEMM via dispatcher
    float exec_time;
    try
    {
        exec_time = g_dispatcher->run(A_dev, B_dev, C_dev, problem);
    }
    catch(const std::exception& e)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Copy result back to host
    if(hipMemcpy(
           C_host, C_dev, static_cast<size_t>(M) * N * sizeof(CDataType), hipMemcpyDeviceToHost) !=
       hipSuccess)
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
 * Kept for backward compatibility with one-kernel-per-.so callers.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Get the name of the kernel at a given registry index (multi-kernel ABI).
 *
 * Mirrors the conv/fmha ctypes libs: copies the index-th registered kernel's
 * name into the caller-provided buffer so one .so can report a whole batch and
 * be selected by name at runtime. Returns 0 on success, -1 on bad args or
 * out-of-range index.
 */
int dispatcher_get_kernel_name_at(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0)
    {
        return -1;
    }

    auto kernels = Registry::instance().get_all();
    if(index < 0 || index >= static_cast<int>(kernels.size()))
    {
        return -1;
    }

    std::string name = kernels[index]->get_name();
    std::strncpy(buffer, name.c_str(), static_cast<size_t>(buffer_size) - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

/**
 * Initialize dispatcher (alias)
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Get the number of registered kernels
 */
int dispatcher_get_kernel_count() { return static_cast<int>(Registry::instance().size()); }

/**
 * Export registry to JSON string
 */
static std::string g_json_buffer;

const char* dispatcher_export_registry_json()
{
    auto& registry = Registry::instance();

    std::ostringstream json;
    json << "{\n";
    json << "  \"metadata\": {\n";
    json << "    \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    json << "    \"total_kernels\": " << registry.size() << ",\n";
    json << "    \"export_version\": \"1.0\",\n";
    json << "    \"dispatcher_version\": \"1.0.0\"\n";
    json << "  },\n";
    json << "  \"statistics\": {\n";
    json << "    \"by_datatype\": {},\n";
    json << "    \"by_pipeline\": {},\n";
    json << "    \"by_scheduler\": {}\n";
    json << "  },\n";
    json << "  \"kernels\": [\n";

    auto kernels = registry.get_all();
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        auto& kernel     = kernels[i];
        auto& key        = kernel->get_key();
        auto& algo       = key.algorithm;
        std::string name = kernel->get_name();

        json << "    {\n";
        json << "      \"identifier\": \"" << key.encode_identifier() << "\",\n";
        json << "      \"name\": \"" << name << "\",\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_shape\": {\"m\": " << algo.tile_shape.m
             << ", \"n\": " << algo.tile_shape.n << ", \"k\": " << algo.tile_shape.k << "},\n";
        json << "        \"wave_shape\": {\"m\": " << unsigned(algo.wave_shape.m)
             << ", \"n\": " << unsigned(algo.wave_shape.n)
             << ", \"k\": " << unsigned(algo.wave_shape.k) << "},\n";
        json << "        \"warp_tile_shape\": {\"m\": " << unsigned(algo.warp_tile_shape.m)
             << ", \"n\": " << unsigned(algo.warp_tile_shape.n)
             << ", \"k\": " << unsigned(algo.warp_tile_shape.k) << "},\n";
        json << "        \"block_size\": " << algo.block_size << ",\n";
        json << "        \"persistent\": " << (algo.persistent ? "true" : "false") << ",\n";
        json << "        \"double_buffer\": " << (algo.double_buffer ? "true" : "false") << ",\n";
        json << "        \"preshuffle\": " << (algo.preshuffle ? "true" : "false") << ",\n";
        json << "        \"transpose_c\": " << (algo.transpose_c ? "true" : "false") << "\n";
        json << "      }\n";
        json << "    }";
        if(i < kernels.size() - 1)
        {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    g_json_buffer = json.str();
    return g_json_buffer.c_str();
}

/**
 * Cleanup dispatcher resources
 */
void dispatcher_cleanup()
{
    g_dispatcher.reset();
    g_initialized = false;
#if defined(GEMM_KEY_DTYPE_A)
    // Release the process-lifetime shuffled-B cache so an embedding library that
    // calls dispatcher_cleanup() frees the held HostTensors instead of leaking
    // them until process exit (the benchmark process never calls cleanup, but a
    // library consumer should get its memory back). Empty for non-preshuffle
    // kernels, where clear() is a no-op.
    g_shuffled_b_cache.clear();
#endif
}

} // extern "C"
