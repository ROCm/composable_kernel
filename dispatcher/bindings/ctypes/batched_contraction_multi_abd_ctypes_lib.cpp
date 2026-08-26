// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Batched Contraction Multiple ABD Dispatcher ctypes Library
 *
 * Provides a C API for Python ctypes integration for the batched_contraction_multi_abd op.
 *
 * WHY A SEPARATE .so (divergent ABI):
 *   batched_contraction_multi_abd takes ARRAYS of A, B, and D device pointers
 *   (NumATensors / NumBTensors / NumDTensors) plus per-tensor multi-dimensional
 *   dimension/stride arrays (not the scalar M/N/K the regular GEMM ABI exposes).
 *   Its launch takes a ck_tile::BatchedContractionMultiABDHostArgs<NumDimG, NumDimM,
 *   NumDimN, NumDimK, NumATensor, NumBTensor, NumDTensor>, so this library bypasses
 *   the name-keyed dispatcher registry and calls SelectedKernel::launch(...)
 *   directly on the force-included kernel -- exactly the divergent-ABI bridge
 *   pattern used for gemm_multi_abd and grouped_gemm.
 *
 *   The kernel header is force-included via the -include compiler flag and defines
 *   SelectedKernel, KERNEL_NAME, and the tensor/dimension counts (inside the
 *   kernel's namespace, re-exported to global scope under CK_TILE_SINGLE_KERNEL_INCLUDE).
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libbatched_contraction_multi_abd_<name>.so")
 *   lib.dispatcher_initialize()
 *   lib.dispatcher_run_batched_contraction_multi_abd(
 *       as_ptrs, bs_ptrs, ds_ptrs, e_ptr,
 *       num_a, num_b, num_d,
 *       g_dims, m_dims, n_dims, k_dims,
 *       num_dim_g, num_dim_m, num_dim_n, num_dim_k,
 *       a_strides_flat, b_strides_flat, d_strides_flat, e_strides,
 *       elem_a, elem_b, elem_d, elem_e,
 *       k_batch, &time_ms)
 *
 * Dimension / stride arrays layout (passed as flat int64 arrays from Python):
 *   g_dims[num_dim_g], m_dims[num_dim_m], n_dims[num_dim_n], k_dims[num_dim_k]
 *   a_strides_flat[num_a * (num_dim_g + num_dim_m + num_dim_k)]
 *   b_strides_flat[num_b * (num_dim_g + num_dim_n + num_dim_k)]
 *   d_strides_flat[num_d * (num_dim_g + num_dim_m + num_dim_n)]
 *   e_strides[num_dim_g + num_dim_m + num_dim_n]
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"

// Kernel header force-included via -include compiler flag.
// Under CK_TILE_SINGLE_KERNEL_INCLUDE it exports to global scope:
//   SelectedKernel  -- the generated struct with static launch()
//   KERNEL_NAME     -- byte-exact runtime kernel name
//   NumATensors, NumBTensors, NumDTensors  -- tensor counts
//   NumDimsG, NumDimsM, NumDimsN, NumDimsK -- dimension counts

#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

// Guard with fallbacks so the file is still self-describing if the macros change.
#ifndef CONTRACTION_MULTI_ABD_NUM_A
#define CONTRACTION_MULTI_ABD_NUM_A NumATensors
#endif
#ifndef CONTRACTION_MULTI_ABD_NUM_B
#define CONTRACTION_MULTI_ABD_NUM_B NumBTensors
#endif
#ifndef CONTRACTION_MULTI_ABD_NUM_D
#define CONTRACTION_MULTI_ABD_NUM_D NumDTensors
#endif
#ifndef CONTRACTION_MULTI_ABD_DIM_G
#define CONTRACTION_MULTI_ABD_DIM_G NumDimsG
#endif
#ifndef CONTRACTION_MULTI_ABD_DIM_M
#define CONTRACTION_MULTI_ABD_DIM_M NumDimsM
#endif
#ifndef CONTRACTION_MULTI_ABD_DIM_N
#define CONTRACTION_MULTI_ABD_DIM_N NumDimsN
#endif
#ifndef CONTRACTION_MULTI_ABD_DIM_K
#define CONTRACTION_MULTI_ABD_DIM_K NumDimsK
#endif

namespace {

constexpr ck_tile::index_t kNumA    = CONTRACTION_MULTI_ABD_NUM_A;
constexpr ck_tile::index_t kNumB    = CONTRACTION_MULTI_ABD_NUM_B;
constexpr ck_tile::index_t kNumD    = CONTRACTION_MULTI_ABD_NUM_D;
constexpr ck_tile::index_t kNumDimG = CONTRACTION_MULTI_ABD_DIM_G;
constexpr ck_tile::index_t kNumDimM = CONTRACTION_MULTI_ABD_DIM_M;
constexpr ck_tile::index_t kNumDimN = CONTRACTION_MULTI_ABD_DIM_N;
constexpr ck_tile::index_t kNumDimK = CONTRACTION_MULTI_ABD_DIM_K;

// Dimension sizes for each tensor's stride/dim arrays
constexpr ck_tile::index_t kADimSize = kNumDimG + kNumDimM + kNumDimK; // [G,M,K]
constexpr ck_tile::index_t kBDimSize = kNumDimG + kNumDimN + kNumDimK; // [G,N,K]
constexpr ck_tile::index_t kEDimSize = kNumDimG + kNumDimM + kNumDimN; // [G,M,N]

using HostArgs = ck_tile::
    BatchedContractionMultiABDHostArgs<kNumDimG, kNumDimM, kNumDimN, kNumDimK, kNumA, kNumB, kNumD>;

using ADims = typename HostArgs::ADims;
using BDims = typename HostArgs::BDims;
using EDims = typename HostArgs::EDims;

/// RAII owner for a hipMalloc'd buffer -- frees on scope exit, including the
/// early-return and exception paths.
struct HipDeleter
{
    void operator()(void* p) const noexcept
    {
        if(p)
            (void)hipFree(p);
    }
};

using DevicePtr = std::unique_ptr<void, HipDeleter>;

/// Allocate `bytes` on the device, returning an owning pointer (null on failure).
DevicePtr device_alloc(size_t bytes)
{
    void* raw = nullptr;
    if(hipMalloc(&raw, bytes) != hipSuccess)
        return DevicePtr{};
    return DevicePtr{raw};
}

/**
 * Number of elements spanned by a strided tensor, i.e. one past the largest
 * addressable offset: 1 + sum_i (dim_i - 1) * stride_i.
 *
 * The kernel indexes each tensor through caller-supplied per-dimension strides,
 * so the product of the logical dimensions understates the buffer for padded,
 * transposed, or otherwise non-packed layouts. Sizing allocations and copies
 * from this span keeps every offset the kernel can form inside the allocation.
 * For a packed row-major tensor this equals the product of its dimensions.
 *
 * Returns -1 if any dimension is non-positive, any stride is negative, or the
 * computation would overflow int64.
 */
int64_t strided_span(const int64_t* dims, const int64_t* strides, int rank)
{
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

    int64_t span = 1;
    for(int i = 0; i < rank; ++i)
    {
        if(dims[i] <= 0 || strides[i] < 0)
            return -1;

        // extent_i = (dim_i - 1) * stride_i, guarded against overflow
        const int64_t extent_terms = dims[i] - 1;
        if(strides[i] != 0 && extent_terms > kMax / strides[i])
            return -1;
        const int64_t extent = extent_terms * strides[i];

        if(span > kMax - extent)
            return -1;
        span += extent;
    }
    return span;
}

bool g_initialized = false;

} // namespace

extern "C" {

/**
 * Initialize the library. Registry-bypass path -- just flips the ready flag.
 * Kept for ABI symmetry with the regular GEMM ctypes lib.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

int dispatcher_init() { return dispatcher_initialize(); }

/** Tensor/dimension counts baked in at compile time. */
int dispatcher_get_num_a_tensors() { return static_cast<int>(kNumA); }
int dispatcher_get_num_b_tensors() { return static_cast<int>(kNumB); }
int dispatcher_get_num_d_tensors() { return static_cast<int>(kNumD); }
int dispatcher_get_num_dim_g() { return static_cast<int>(kNumDimG); }
int dispatcher_get_num_dim_m() { return static_cast<int>(kNumDimM); }
int dispatcher_get_num_dim_n() { return static_cast<int>(kNumDimN); }
int dispatcher_get_num_dim_k() { return static_cast<int>(kNumDimK); }

/** Byte-exact kernel name baked into the force-included header. */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

int dispatcher_get_kernel_name_at(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0 || index != 0)
        return -1;
    std::strncpy(buffer, KERNEL_NAME, static_cast<size_t>(buffer_size) - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

int dispatcher_get_kernel_count() { return 1; }

/**
 * Run a batched contraction multi-ABD on the GPU.
 *
 * All tensors are passed as host pointers (contiguous numpy arrays from Python).
 * This shim owns the GPU side: allocates device buffers, uploads inputs, launches,
 * and downloads E.
 *
 * Dimension/stride arrays are flat int64 arrays:
 *   g_dims[num_dim_g], m_dims[num_dim_m], n_dims[num_dim_n], k_dims[num_dim_k]
 *   a_strides_flat[num_a * (G+M+K)] -- row-major: first tensor's dims, then second, ...
 *   b_strides_flat[num_b * (G+N+K)]
 *   d_strides_flat[num_d * (G+M+N)]
 *   e_strides[G+M+N]
 *
 * Returns 0 on success, -1 on HIP/arg error, -2 on unsupported args, -3 on count mismatch.
 */
int dispatcher_run_batched_contraction_multi_abd(const void** as_hosts,
                                                 const void** bs_hosts,
                                                 const void** ds_hosts,
                                                 void* e_host,
                                                 int num_a,
                                                 int num_b,
                                                 int num_d,
                                                 const int64_t* g_dims,
                                                 const int64_t* m_dims,
                                                 const int64_t* n_dims,
                                                 const int64_t* k_dims,
                                                 int num_dim_g,
                                                 int num_dim_m,
                                                 int num_dim_n,
                                                 int num_dim_k,
                                                 const int64_t* a_strides_flat,
                                                 const int64_t* b_strides_flat,
                                                 const int64_t* d_strides_flat,
                                                 const int64_t* e_strides,
                                                 int elem_a,
                                                 int elem_b,
                                                 int elem_d,
                                                 int elem_e,
                                                 int64_t k_batch,
                                                 float* time_ms)
{
    if(!g_initialized || !as_hosts || !bs_hosts || !e_host)
        return -1;
    if(!g_dims || !m_dims || !n_dims || !k_dims)
        return -1;
    if(!a_strides_flat || !b_strides_flat || !e_strides)
        return -1;
    if(elem_a <= 0 || elem_b <= 0 || elem_e <= 0)
        return -1;
    // elem_d is only meaningful when there are D tensors; skip the check otherwise
    // so callers with kNumD==0 can safely pass elem_d=0 without being incorrectly rejected.

    // Tensor/dimension counts must match what's baked into the kernel type.
    if(num_a != static_cast<int>(kNumA) || num_b != static_cast<int>(kNumB) ||
       num_d != static_cast<int>(kNumD))
    {
        std::cerr << "dispatcher_run_batched_contraction_multi_abd: tensor count mismatch "
                  << "(expected num_a=" << kNumA << " num_b=" << kNumB << " num_d=" << kNumD
                  << ", got num_a=" << num_a << " num_b=" << num_b << " num_d=" << num_d << ")\n";
        return -3;
    }
    if(num_dim_g != static_cast<int>(kNumDimG) || num_dim_m != static_cast<int>(kNumDimM) ||
       num_dim_n != static_cast<int>(kNumDimN) || num_dim_k != static_cast<int>(kNumDimK))
    {
        std::cerr << "dispatcher_run_batched_contraction_multi_abd: dimension count mismatch "
                  << "(kernel was compiled with G=" << kNumDimG << " M=" << kNumDimM
                  << " N=" << kNumDimN << " K=" << kNumDimK << ")\n";
        return -3;
    }
    if(kNumD > 0 && (!ds_hosts || !d_strides_flat || elem_d <= 0))
        return -1;

    // split-K is not forwarded to the inner kernel (HostArgs has no k_batch field).
    // Reject non-default values so the caller never silently gets k_batch=1 behavior.
    if(k_batch != 1)
    {
        std::cerr << "dispatcher_run_batched_contraction_multi_abd: split-K (k_batch=" << k_batch
                  << ") is not supported; only k_batch=1 is accepted.\n";
        return -2;
    }

    // Guard: every dimension and stride must fit in int32 (ck_tile::index_t).
    // Check all flat arrays before any allocation or cast to avoid silent truncation.
    {
        auto check_range = [](const int64_t* arr, int len, const char* name) -> bool {
            constexpr int64_t kMaxIdx =
                static_cast<int64_t>(std::numeric_limits<ck_tile::index_t>::max());
            for(int i = 0; i < len; ++i)
            {
                if(arr[i] < 0 || arr[i] > kMaxIdx)
                {
                    std::cerr << "dispatcher_run_batched_contraction_multi_abd: " << name << "["
                              << i << "]=" << arr[i] << " overflows ck_tile::index_t (int32).\n";
                    return false;
                }
            }
            return true;
        };

        if(!check_range(g_dims, num_dim_g, "g_dims") || !check_range(m_dims, num_dim_m, "m_dims") ||
           !check_range(n_dims, num_dim_n, "n_dims") || !check_range(k_dims, num_dim_k, "k_dims"))
            return -1;

        if(!check_range(a_strides_flat, num_a * static_cast<int>(kADimSize), "a_strides_flat") ||
           !check_range(b_strides_flat, num_b * static_cast<int>(kBDimSize), "b_strides_flat") ||
           !check_range(e_strides, static_cast<int>(kEDimSize), "e_strides"))
            return -1;

        if(kNumD > 0 && d_strides_flat)
        {
            if(!check_range(d_strides_flat, num_d * static_cast<int>(kEDimSize), "d_strides_flat"))
                return -1;
        }
    }

    // Per-tensor logical dimensions, concatenated in the order the kernel indexes them.
    //   A: [G..., M..., K...]   B: [G..., N..., K...]   D/E: [G..., M..., N...]
    auto concat_dims = [](const int64_t* first,
                          int n_first,
                          const int64_t* second,
                          int n_second,
                          const int64_t* third,
                          int n_third) {
        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(n_first + n_second + n_third));
        out.insert(out.end(), first, first + n_first);
        out.insert(out.end(), second, second + n_second);
        out.insert(out.end(), third, third + n_third);
        return out;
    };

    const std::vector<int64_t> a_dims_cat =
        concat_dims(g_dims, num_dim_g, m_dims, num_dim_m, k_dims, num_dim_k);
    const std::vector<int64_t> b_dims_cat =
        concat_dims(g_dims, num_dim_g, n_dims, num_dim_n, k_dims, num_dim_k);
    const std::vector<int64_t> e_dims_cat =
        concat_dims(g_dims, num_dim_g, m_dims, num_dim_m, n_dims, num_dim_n);

    // Size every device buffer from the tensor's addressable span rather than the
    // product of its dimensions: the kernel walks each tensor through the strides
    // supplied by the caller, so a padded or transposed layout can reach past the
    // dimension product. For packed layouts the two are equal.
    //
    // The caller must guarantee each host buffer holds at least this many elements;
    // that is enforced Python-side (contiguity + element-count checks) before the
    // pointers reach this ABI.
    auto span_bytes = [](const std::vector<int64_t>& dims,
                         const int64_t* strides,
                         int elem_size,
                         const char* name,
                         size_t& out) -> bool {
        const int64_t span = strided_span(dims.data(), strides, static_cast<int>(dims.size()));
        if(span < 0)
        {
            std::cerr << "dispatcher_run_batched_contraction_multi_abd: " << name
                      << " has an invalid or overflowing dimension/stride combination.\n";
            return false;
        }
        out = static_cast<size_t>(span) * static_cast<size_t>(elem_size);
        return true;
    };

    std::vector<size_t> a_bytes(kNumA, 0);
    std::vector<size_t> b_bytes(kNumB, 0);
    std::vector<size_t> d_bytes(kNumD, 0);
    size_t e_bytes = 0;

    for(int i = 0; i < num_a; ++i)
    {
        if(!span_bytes(a_dims_cat, a_strides_flat + i * kADimSize, elem_a, "A", a_bytes[i]))
            return -1;
    }
    for(int i = 0; i < num_b; ++i)
    {
        if(!span_bytes(b_dims_cat, b_strides_flat + i * kBDimSize, elem_b, "B", b_bytes[i]))
            return -1;
    }
    for(int i = 0; i < num_d; ++i)
    {
        if(!span_bytes(e_dims_cat, d_strides_flat + i * kEDimSize, elem_d, "D", d_bytes[i]))
            return -1;
    }
    if(!span_bytes(e_dims_cat, e_strides, elem_e, "E", e_bytes))
        return -1;

    // Owning device buffers -- freed automatically on every return path below.
    std::vector<DevicePtr> a_dev(kNumA);
    std::vector<DevicePtr> b_dev(kNumB);
    std::vector<DevicePtr> d_dev(kNumD);
    DevicePtr e_dev;

    for(int i = 0; i < num_a; ++i)
    {
        a_dev[i] = device_alloc(a_bytes[i]);
        if(!a_dev[i] ||
           hipMemcpy(a_dev[i].get(), as_hosts[i], a_bytes[i], hipMemcpyHostToDevice) != hipSuccess)
            return -1;
    }
    for(int i = 0; i < num_b; ++i)
    {
        b_dev[i] = device_alloc(b_bytes[i]);
        if(!b_dev[i] ||
           hipMemcpy(b_dev[i].get(), bs_hosts[i], b_bytes[i], hipMemcpyHostToDevice) != hipSuccess)
            return -1;
    }
    for(int i = 0; i < num_d; ++i)
    {
        d_dev[i] = device_alloc(d_bytes[i]);
        if(!d_dev[i] ||
           hipMemcpy(d_dev[i].get(), ds_hosts[i], d_bytes[i], hipMemcpyHostToDevice) != hipSuccess)
            return -1;
    }
    e_dev = device_alloc(e_bytes);
    if(!e_dev || hipMemset(e_dev.get(), 0, e_bytes) != hipSuccess)
        return -1;

    // Build std::array<ADims, NumATensor> for as_ptr, As_dims, As_strides, etc.
    std::array<const void*, kNumA> as_dev{};
    std::array<const void*, kNumB> bs_dev{};
    std::array<const void*, kNumD> ds_dev{};

    std::array<ADims, kNumA> As_dims{};
    std::array<BDims, kNumB> Bs_dims{};
    std::array<EDims, kNumD> Ds_dims{};
    EDims E_dims_arr{};

    std::array<ADims, kNumA> As_strides{};
    std::array<BDims, kNumB> Bs_strides{};
    std::array<EDims, kNumD> Ds_strides{};
    EDims E_strides_arr{};

    for(ck_tile::index_t i = 0; i < kNumA; ++i)
        as_dev[i] = a_dev[i].get();
    for(ck_tile::index_t i = 0; i < kNumB; ++i)
        bs_dev[i] = b_dev[i].get();
    for(ck_tile::index_t i = 0; i < kNumD; ++i)
        ds_dev[i] = d_dev[i].get();

    // Fill dimension and stride arrays from flat caller arrays.
    // A dims layout: [G0,..., M0,..., K0,...]
    for(ck_tile::index_t a = 0; a < kNumA; ++a)
    {
        const int64_t* a_strides = a_strides_flat + a * kADimSize;
        int pos                  = 0;
        for(int g = 0; g < num_dim_g; ++g, ++pos)
        {
            As_dims[a][pos]    = static_cast<ck_tile::index_t>(g_dims[g]);
            As_strides[a][pos] = static_cast<ck_tile::index_t>(a_strides[pos]);
        }
        for(int m = 0; m < num_dim_m; ++m, ++pos)
        {
            As_dims[a][pos]    = static_cast<ck_tile::index_t>(m_dims[m]);
            As_strides[a][pos] = static_cast<ck_tile::index_t>(a_strides[pos]);
        }
        for(int k = 0; k < num_dim_k; ++k, ++pos)
        {
            As_dims[a][pos]    = static_cast<ck_tile::index_t>(k_dims[k]);
            As_strides[a][pos] = static_cast<ck_tile::index_t>(a_strides[pos]);
        }
    }

    // B dims layout: [G0,..., N0,..., K0,...]
    for(ck_tile::index_t b = 0; b < kNumB; ++b)
    {
        const int64_t* b_strides = b_strides_flat + b * kBDimSize;
        int pos                  = 0;
        for(int g = 0; g < num_dim_g; ++g, ++pos)
        {
            Bs_dims[b][pos]    = static_cast<ck_tile::index_t>(g_dims[g]);
            Bs_strides[b][pos] = static_cast<ck_tile::index_t>(b_strides[pos]);
        }
        for(int n = 0; n < num_dim_n; ++n, ++pos)
        {
            Bs_dims[b][pos]    = static_cast<ck_tile::index_t>(n_dims[n]);
            Bs_strides[b][pos] = static_cast<ck_tile::index_t>(b_strides[pos]);
        }
        for(int k = 0; k < num_dim_k; ++k, ++pos)
        {
            Bs_dims[b][pos]    = static_cast<ck_tile::index_t>(k_dims[k]);
            Bs_strides[b][pos] = static_cast<ck_tile::index_t>(b_strides[pos]);
        }
    }

    // D and E dims layout: [G0,..., M0,..., N0,...]
    {
        int pos = 0;
        for(int g = 0; g < num_dim_g; ++g, ++pos)
        {
            E_dims_arr[pos]    = static_cast<ck_tile::index_t>(g_dims[g]);
            E_strides_arr[pos] = static_cast<ck_tile::index_t>(e_strides[pos]);
        }
        for(int m = 0; m < num_dim_m; ++m, ++pos)
        {
            E_dims_arr[pos]    = static_cast<ck_tile::index_t>(m_dims[m]);
            E_strides_arr[pos] = static_cast<ck_tile::index_t>(e_strides[pos]);
        }
        for(int n = 0; n < num_dim_n; ++n, ++pos)
        {
            E_dims_arr[pos]    = static_cast<ck_tile::index_t>(n_dims[n]);
            E_strides_arr[pos] = static_cast<ck_tile::index_t>(e_strides[pos]);
        }
    }

    for(ck_tile::index_t d = 0; d < kNumD; ++d)
    {
        const int64_t* d_strides = d_strides_flat + d * kEDimSize;
        int pos                  = 0;
        for(int g = 0; g < num_dim_g; ++g, ++pos)
        {
            Ds_dims[d][pos]    = E_dims_arr[g];
            Ds_strides[d][pos] = static_cast<ck_tile::index_t>(d_strides[pos]);
        }
        for(int m = 0; m < num_dim_m; ++m, ++pos)
        {
            Ds_dims[d][pos]    = E_dims_arr[num_dim_g + m];
            Ds_strides[d][pos] = static_cast<ck_tile::index_t>(d_strides[pos]);
        }
        for(int n = 0; n < num_dim_n; ++n, ++pos)
        {
            Ds_dims[d][pos]    = E_dims_arr[num_dim_g + num_dim_m + n];
            Ds_strides[d][pos] = static_cast<ck_tile::index_t>(d_strides[pos]);
        }
    }

    HostArgs host_args{as_dev,
                       bs_dev,
                       ds_dev,
                       e_dev.get(),
                       As_dims,
                       Bs_dims,
                       Ds_dims,
                       E_dims_arr,
                       As_strides,
                       Bs_strides,
                       Ds_strides,
                       E_strides_arr};

    float exec_time = -1.0f;
    try
    {
        ck_tile::stream_config stream{nullptr, /*time_kernel=*/true};
        exec_time = SelectedKernel::launch(host_args, stream);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "batched_contraction_multi_abd launch failed: " << ex.what() << std::endl;
        if(time_ms)
            *time_ms = -1.0f;
        return -2;
    }

    if(exec_time < 0.0f)
    {
        // IsSupportedArguments returned false inside launch
        if(time_ms)
            *time_ms = -1.0f;
        return -2;
    }

    if(hipMemcpy(e_host, e_dev.get(), e_bytes, hipMemcpyDeviceToHost) != hipSuccess)
        return -1;

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
