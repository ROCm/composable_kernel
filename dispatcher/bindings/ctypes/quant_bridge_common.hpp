// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Shared infrastructure for the block-scale quant GEMM ctypes bridges.
 *
 * The five per-op bridges (tensor_quant, rowcolquant, aquant, abquant, bquant)
 * each compile one kernel per .so, force-including the generated kernel header:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_<op>_ctypes_lib.cpp
 *
 * Because every bridge is its own translation unit, this common layer is
 * header-only: the templates / inline helpers below each get their own copy per
 * .so, so there is no shared library and no ODR concern. It centralizes the
 * infrastructure that used to be copy-pasted into all five sources -- device
 * memory management, GPU-arch validation, the entry guard chain, quant-group
 * validation, kernel-launch timing, init/cleanup, and the exported C
 * boilerplate -- so a fix to any of those happens exactly once.
 *
 * run_scalar_quant_gemm() goes one step further and holds the *entire* run()
 * body for the two bridges (tensor_quant, rowcolquant) that neither reshuffle
 * their operands nor carry a quant group size; those two sources are reduced to
 * an entry point plus their scale extents.
 *
 * The three bridges that do reshuffle (aquant, abquant, bquant) keep their own
 * argument construction here, and their host-side reshuffle steps live in
 * quant_bridge_shuffle.hpp.
 *
 * The generated kernel header (force-included before this one) must already
 * provide ck_tile (numeric_traits, stream_config, QuantGemmHostArgs, index_t)
 * and the KERNEL_NAME macro.
 */

#ifndef CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
#define CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>

// On a HIP error, print the failing op name + file/line and return -1. RAII
// DeviceBuffers free themselves on the return, so no cleanup call is needed.
// Defined before the namespace because the helpers below use it too.
#define BRIDGE_HIP_CHECK(fn, call)                                                                \
    do                                                                                            \
    {                                                                                             \
        hipError_t _err = (call);                                                                 \
        if(_err != hipSuccess)                                                                    \
        {                                                                                         \
            std::cerr << (fn) << ": HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ \
                      << ":" << __LINE__ << "\n";                                                 \
            return -1;                                                                            \
        }                                                                                         \
    } while(0)

// Emit the C API boilerplate shared by every bridge. Invoke once at the top of
// each op's `extern "C"` block (it declares the file-local g_initialized flag
// that the op's run() guard checks). KERNEL_NAME is force-included.
//
// Threading/lifecycle contract: g_initialized is a plain (non-atomic) file-local
// flag. dispatcher_initialize()/dispatcher_cleanup()/run() are NOT synchronized;
// this ABI is intended for single-threaded use (the Python ctypes harness).
// Callers must initialize before run() and must not invoke these entry points
// concurrently across threads.
#define QUANT_BRIDGE_C_API()                                         \
    static bool g_initialized = false;                               \
    int dispatcher_initialize()                                      \
    {                                                                \
        g_initialized = true;                                        \
        return 0;                                                    \
    }                                                                \
    const char* dispatcher_get_kernel_name() { return KERNEL_NAME; } \
    int dispatcher_init() { return dispatcher_initialize(); }        \
    int dispatcher_get_kernel_count() { return 1; }                  \
    void dispatcher_cleanup() { g_initialized = false; }

namespace quant_bridge {

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// RAII owner for a device allocation. Frees on scope exit, which removes the
// hand-written `cleanup` lambda that every bridge used to duplicate: any early
// return (including from BRIDGE_HIP_CHECK) releases every buffer automatically.
template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;

    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }

    // Allocate `bytes` bytes; returns the hipMalloc status for BRIDGE_HIP_CHECK.
    hipError_t allocate(std::size_t bytes) { return hipMalloc(&ptr, bytes); }

    operator T*() const { return ptr; }
};

// Derive the GPU architecture from the running device (never assume one at
// compile time) and reject unsupported archs. gfx942 and gfx950 are always
// accepted; gfx90a is accepted only when the op supports it (aquant, bquant).
inline bool validate_supported_arch(const char* fn, bool allow_gfx90a = false)
{
    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << fn << ": could not query device architecture\n";
        return false;
    }
    const std::string arch(props.gcnArchName);
    const bool ok = arch.rfind("gfx950", 0) == 0 || arch.rfind("gfx942", 0) == 0 ||
                    arch.rfind("gfx1250", 0) == 0 || (allow_gfx90a && arch.rfind("gfx90a", 0) == 0);
    if(!ok)
    {
        std::cerr << fn << ": unsupported GPU architecture '" << arch
                  << "' (supported: " << (allow_gfx90a ? "gfx90a, " : "")
                  << "gfx942, gfx950, gfx1250)\n";
        return false;
    }
    return true;
}

// Build the stream_config used for a launch. When timing is requested use the
// GPU timer with warmup (cold_niters=3, nrepeat=10); otherwise run once.
inline ck_tile::stream_config make_stream_config(bool do_time)
{
    return ck_tile::stream_config{
        nullptr,          // stream_id_
        do_time,          // time_kernel_
        0,                // log_level_
        do_time ? 3 : 0,  // cold_niters_
        do_time ? 10 : 1, // nrepeat_
        do_time,          // is_gpu_timer_
        false,            // flush_cache_
        1,                // rotating_count_
    };
}

// Direct-launch the force-included kernel. Returns the kernel execution time in
// ms, or a negative value if the kernel reports unsupported args (callers treat
// <0 as an error and return -2, matching the previous behavior).
template <typename KernelT>
inline float launch(const ck_tile::QuantGemmHostArgs& args, bool do_time)
{
    return KernelT::launch(args, make_stream_config(do_time));
}

// The three guard helpers below replace the init / null-pointer / dimension
// checks that every bridge's run() used to inline verbatim. Each prints the same
// diagnostic as before and returns false so the caller can `return -1`. The
// per-op argument lists differ (4 vs 5 pointers; MNK plus op-specific QK/QN
// counts), so the pointer/dimension checks take an initializer_list.
inline bool check_initialized(const char* fn, bool initialized)
{
    if(!initialized)
    {
        std::cerr << fn << ": not initialized\n";
        return false;
    }
    return true;
}

inline bool check_non_null(const char* fn, std::initializer_list<const void*> ptrs)
{
    for(const void* p : ptrs)
    {
        if(!p)
        {
            std::cerr << fn << ": null pointer argument\n";
            return false;
        }
    }
    return true;
}

inline bool check_positive_dims(const char* fn, std::initializer_list<int64_t> dims)
{
    for(int64_t d : dims)
    {
        if(d <= 0)
        {
            std::cerr << fn << ": invalid dimensions\n";
            return false;
        }
    }
    return true;
}

// The entry guard every bridge opens with, in the order they all used: init flag,
// null pointers, positive dimensions, then GPU arch. Returns false once anything
// fails (each helper has already printed its own diagnostic) so the caller can
// `return -1`.
//
// check_arch exists for abquant, which must run its compile-time fp4-preshuffle
// reject (return -3) between the argument checks and the arch check; it passes
// false here and calls validate_supported_arch() itself afterwards.
inline bool check_entry_args(const char* fn,
                             bool initialized,
                             std::initializer_list<const void*> ptrs,
                             std::initializer_list<int64_t> dims,
                             bool allow_gfx90a = false,
                             bool check_arch   = true)
{
    return check_initialized(fn, initialized) && check_non_null(fn, ptrs) &&
           check_positive_dims(fn, dims) &&
           (!check_arch || validate_supported_arch(fn, allow_gfx90a));
}

// Verify one caller-supplied scale count against the quant group size baked into
// this .so: `count` must equal ceil(dim / group). A mismatch means the host built
// its scale tensor for a different group size than the compiled kernel reads, so
// the kernel would index past the end of it.
inline bool check_quant_group_count(const char* fn,
                                    const char* count_name,
                                    int64_t count,
                                    const char* dim_name,
                                    int64_t dim,
                                    int64_t group)
{
    const int64_t expected = (dim + group - 1) / group;
    if(count == expected)
        return true;
    std::cerr << fn << ": " << count_name << " mismatch. Got " << count << ", expected " << expected
              << " for " << dim_name << "=" << dim << " with quant group size " << group << "\n";
    return false;
}

// The identical tail every bridge's run() ended with: direct-launch the
// force-included kernel (return -2 if it rejects the args), copy C back to the
// host, publish the optional timing, return 0. C_dev is any type convertible to
// const CT* (e.g. DeviceBuffer<CT>).
template <typename KernelT, typename CT>
inline int launch_and_copyback(const char* fn,
                               const ck_tile::QuantGemmHostArgs& args,
                               void* C_host,
                               const CT* C_dev,
                               std::size_t mn_elems,
                               float* time_ms)
{
    const float exec_time = launch<KernelT>(args, time_ms != nullptr);
    if(exec_time < 0.0f)
    {
        std::cerr << fn << ": kernel reported unsupported args\n";
        return -2;
    }

    const hipError_t err =
        hipMemcpy(C_host, C_dev, elements_to_bytes<CT>(mn_elems), hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        std::cerr << fn << ": HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":"
                  << __LINE__ << "\n";
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

// The complete run() body shared by the tensor_quant and rowcolquant bridges.
// Neither reshuffles anything and neither has a quant group size, so both reduce
// to: guard, require packed strides, copy five buffers up, launch, copy C back.
// They differ only in how many scale elements each side carries -- tensor_quant
// passes one scalar per tensor (aq_elems = bq_elems = 1), rowcolquant one per A
// row / B column (M and N) -- which is why those are runtime arguments rather
// than another pair of template parameters.
//
// QK_A/QK_B and stride_AQ/stride_BQ are hardcoded to 1 for both: neither kernel
// has quant groups, and both index their scales by position rather than by a
// scale stride (mirrors the TensorQuant / RowColQuant branches of
// run_gemm_quant_example.inc).
template <typename KernelT, typename AT, typename BT, typename CT, typename QT>
inline int run_scalar_quant_gemm(const char* fn,
                                 bool initialized,
                                 const void* A,
                                 const void* B,
                                 const void* AQ,
                                 const void* BQ,
                                 void* C,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 int64_t stride_A,
                                 int64_t stride_B,
                                 int64_t stride_C,
                                 std::size_t aq_elems,
                                 std::size_t bq_elems,
                                 int k_batch,
                                 float* time_ms)
{
    if(!check_entry_args(fn, initialized, {A, B, AQ, BQ, C}, {M, N, K}))
        return -1;

    // Only packed (contiguous) layouts are supported: A is [M,K] row-major, B is
    // [K,N] column-major (leading dim K), C is [M,N] row-major.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << fn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_C=" << N << ", got stride_A=" << stride_A
                  << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
        return -1;
    }

    DeviceBuffer<AT> A_dev;
    DeviceBuffer<BT> B_dev;
    DeviceBuffer<QT> AQ_dev;
    DeviceBuffer<QT> BQ_dev;
    DeviceBuffer<CT> C_dev;
    BRIDGE_HIP_CHECK(fn, A_dev.allocate(elements_to_bytes<AT>(M * K)));
    BRIDGE_HIP_CHECK(fn, B_dev.allocate(elements_to_bytes<BT>(K * N)));
    BRIDGE_HIP_CHECK(fn, AQ_dev.allocate(elements_to_bytes<QT>(aq_elems)));
    BRIDGE_HIP_CHECK(fn, BQ_dev.allocate(elements_to_bytes<QT>(bq_elems)));
    BRIDGE_HIP_CHECK(fn, C_dev.allocate(elements_to_bytes<CT>(M * N)));

    BRIDGE_HIP_CHECK(fn, hipMemcpy(A_dev, A, elements_to_bytes<AT>(M * K), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemcpy(B_dev, B, elements_to_bytes<BT>(K * N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn,
                     hipMemcpy(AQ_dev, AQ, elements_to_bytes<QT>(aq_elems), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn,
                     hipMemcpy(BQ_dev, BQ, elements_to_bytes<QT>(bq_elems), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(fn, hipMemset(C_dev, 0, elements_to_bytes<CT>(M * N)));

    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = 1;
    args.QK_B      = 1;
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 1;
    args.stride_BQ = 1;

    return launch_and_copyback<KernelT, CT>(
        fn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // namespace quant_bridge

#endif // CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
