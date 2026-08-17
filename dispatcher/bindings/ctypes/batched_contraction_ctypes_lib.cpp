// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Batched-Contraction Dispatcher ctypes Library (TileEngine -> Dispatcher bridge).
 *
 * Provides a C API for Python ctypes integration. The kernel header is
 * force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE batched_contraction_ctypes_lib.cpp
 *
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * EDataType, AccDataType, CONTRACTION_KEY_NUM_DIM_{G,M,N,K}, NUM_D_TENSORS.
 *
 * Registry bypass: batched contraction's launch takes
 * ck_tile::BatchedContractionHostArgs<NumDTensor> (variable-length dim/stride
 * vectors), which the generic dispatcher backend cannot express. So this lib
 * builds the HostArgs from plain C arrays and calls SelectedKernel::launch()
 * directly -- the same direct-launch pattern used by the batched/multi-D bridges.
 *
 * Memory model: host-pointer in. The lib owns hipMalloc/hipMemcpy/hipFree.
 * Layouts A=[G..,M..,K..], B=[G..,N..,K..], E=[G..,M..,N..], packed row-major
 * strides (matches the Old-TE profiler's HostTensorDescriptor(dims)).
 * D tensors (num_d>0) share the E shape [G..,M..,N..] and the DBaseDataType
 * element type from codegen; the MultiDAdd/MultiDMultiply epilogue consumes them.
 * split-K (k_batch>1) is rejected: the batched-contraction CShuffle epilogue is
 * hard-wired to memory_operation_enum::set (no atomic accumulation across the
 * blockIdx.z K-splits), so k_batch>1 races and is silently wrong in Old-TE too.
 */

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Kernel header force-included via -include. Brings ck_tile core + the
// ck_tile::BatchedContractionHostArgs type and SelectedKernel/KERNEL_NAME.

#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

static bool g_initialized = false;

namespace {

// Packed row-major strides for a dimension list (matches HostTensorDescriptor).
std::vector<ck_tile::index_t> packed_row_major_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    return strides;
}

int64_t product(const std::vector<ck_tile::index_t>& dims)
{
    int64_t p = 1;
    for(auto d : dims)
        p *= static_cast<int64_t>(d);
    return p;
}

int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if(!v)
        return fallback;
    return std::atoi(v);
}

} // namespace

extern "C" {

int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}
int dispatcher_init() { return dispatcher_initialize(); }

const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }
int dispatcher_get_kernel_count() { return 1; }
int dispatcher_get_num_dim_g() { return CONTRACTION_KEY_NUM_DIM_G; }
int dispatcher_get_num_dim_m() { return CONTRACTION_KEY_NUM_DIM_M; }
int dispatcher_get_num_dim_n() { return CONTRACTION_KEY_NUM_DIM_N; }
int dispatcher_get_num_dim_k() { return CONTRACTION_KEY_NUM_DIM_K; }
int dispatcher_get_num_d_tensors() { return CONTRACTION_KEY_NUM_D_TENSORS; }

void dispatcher_cleanup() { g_initialized = false; }

/**
 * Run batched contraction: E[G..,M..,N..] = sum_K A[G..,M..,K..] * B[G..,N..,K..].
 *
 * A, B, E are host pointers (row-major packed). g/m/n/k_dims give the per-group
 * dimension lengths; their counts must equal the compiled-in NUM_DIM_{G,M,N,K}.
 * Returns 0 ok, -1 HIP/bad-args, -2 kernel reports unsupported args.
 */
int dispatcher_run_batched_contraction(const void* A,
                                       const void* B,
                                       void* E,
                                       const void** d_ptrs,
                                       int num_d,
                                       const int64_t* g_dims,
                                       const int64_t* m_dims,
                                       const int64_t* n_dims,
                                       const int64_t* k_dims,
                                       int num_dim_g,
                                       int num_dim_m,
                                       int num_dim_n,
                                       int num_dim_k,
                                       int k_batch,
                                       float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_batched_contraction: not initialized\n";
        return -1;
    }
    if(!A || !B || !E)
    {
        std::cerr << "dispatcher_run_batched_contraction: null pointer\n";
        return -1;
    }
    constexpr int kNumD = CONTRACTION_KEY_NUM_D_TENSORS;
    if(num_d != kNumD)
    {
        std::cerr << "dispatcher_run_batched_contraction: num_d mismatch, got " << num_d
                  << " compiled " << kNumD << "\n";
        return -1;
    }
    if(kNumD > 0 && !d_ptrs)
    {
        std::cerr << "dispatcher_run_batched_contraction: null d_ptrs with num_d>0\n";
        return -1;
    }
    if(k_batch != 1)
    {
        // split-K (k_batch > 1) is a SHARED Old-TE kernel defect, not a bridge gap:
        // the batched-contraction CShuffle epilogue is hard-wired to
        // memory_operation_enum::set (no atomic accumulation), yet GridSize launches
        // k_batch blockIdx.z K-split blocks that all write the same E tile. Driving
        // this kernel (the exact one Old-TE compiles) at k_batch=2 faults with an
        // illegal memory access on gfx950; k_batch=1 is correct (max_rel ~4e-4). We
        // therefore hard-reject rather than return silently-wrong / crashing results.
        std::cerr << "dispatcher_run_batched_contraction: only k_batch==1 is supported "
                     "(split-K is broken in the shared Old-TE kernel), got "
                  << k_batch << "\n";
        return -1;
    }
    if(num_dim_g <= 0 || num_dim_m <= 0 || num_dim_n <= 0 || num_dim_k <= 0)
    {
        std::cerr << "dispatcher_run_batched_contraction: num_dim_* must be > 0\n";
        return -1;
    }
    if(!g_dims || !m_dims || !n_dims || !k_dims)
    {
        std::cerr << "dispatcher_run_batched_contraction: null dim pointer\n";
        return -1;
    }
    // Every individual dimension length must be strictly positive before we
    // dereference the dim arrays to build the HostArgs shape/stride vectors.
    auto all_positive = [](const int64_t* p, int n) {
        for(int i = 0; i < n; ++i)
            if(p[i] <= 0)
                return false;
        return true;
    };
    if(!all_positive(g_dims, num_dim_g) || !all_positive(m_dims, num_dim_m) ||
       !all_positive(n_dims, num_dim_n) || !all_positive(k_dims, num_dim_k))
    {
        std::cerr << "dispatcher_run_batched_contraction: all dim values must be > 0\n";
        return -1;
    }
    if(num_dim_g != CONTRACTION_KEY_NUM_DIM_G || num_dim_m != CONTRACTION_KEY_NUM_DIM_M ||
       num_dim_n != CONTRACTION_KEY_NUM_DIM_N || num_dim_k != CONTRACTION_KEY_NUM_DIM_K)
    {
        std::cerr << "dispatcher_run_batched_contraction: num_dim mismatch. got (g,m,n,k)=("
                  << num_dim_g << "," << num_dim_m << "," << num_dim_n << "," << num_dim_k
                  << "), compiled (" << CONTRACTION_KEY_NUM_DIM_G << ","
                  << CONTRACTION_KEY_NUM_DIM_M << "," << CONTRACTION_KEY_NUM_DIM_N << ","
                  << CONTRACTION_KEY_NUM_DIM_K << ")\n";
        return -1;
    }

    auto to_vec = [](const int64_t* p, int n) {
        std::vector<ck_tile::index_t> v(n);
        for(int i = 0; i < n; ++i)
            v[i] = static_cast<ck_tile::index_t>(p[i]);
        return v;
    };
    std::vector<ck_tile::index_t> gd = to_vec(g_dims, num_dim_g);
    std::vector<ck_tile::index_t> md = to_vec(m_dims, num_dim_m);
    std::vector<ck_tile::index_t> nd = to_vec(n_dims, num_dim_n);
    std::vector<ck_tile::index_t> kd = to_vec(k_dims, num_dim_k);

    auto concat = [](std::vector<ck_tile::index_t> a,
                     const std::vector<ck_tile::index_t>& b,
                     const std::vector<ck_tile::index_t>& c) {
        a.insert(a.end(), b.begin(), b.end());
        a.insert(a.end(), c.begin(), c.end());
        return a;
    };
    std::vector<ck_tile::index_t> A_dims = concat(gd, md, kd); // [G..,M..,K..]
    std::vector<ck_tile::index_t> B_dims = concat(gd, nd, kd); // [G..,N..,K..]
    std::vector<ck_tile::index_t> E_dims = concat(gd, md, nd); // [G..,M..,N..]

    std::vector<ck_tile::index_t> A_strides = packed_row_major_strides(A_dims);
    std::vector<ck_tile::index_t> B_strides = packed_row_major_strides(B_dims);
    std::vector<ck_tile::index_t> E_strides = packed_row_major_strides(E_dims);

    const int64_t a_elems = product(A_dims);
    const int64_t b_elems = product(B_dims);
    const int64_t e_elems = product(E_dims);
    if(a_elems <= 0 || b_elems <= 0 || e_elems <= 0)
    {
        std::cerr << "dispatcher_run_batched_contraction: non-positive dimension product\n";
        return -1;
    }

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    EDataType* E_host       = static_cast<EDataType*>(E);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    EDataType* E_dev = nullptr;
    // D tensors carry the codegen DBaseDataType element type and the E shape
    // [G..,M..,N..]. Key the device byte-sizing off DBaseDataType (not ADataType) so
    // a future D-dtype divergence from A cannot silently under/over-allocate.
    std::array<DBaseDataType*, kNumD> D_dev{};
    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(E_dev)
            (void)hipFree(E_dev);
        for(int i = 0; i < kNumD; ++i)
            if(D_dev[i])
                (void)hipFree(D_dev[i]);
    };

    if(hipMalloc(&A_dev, a_elems * sizeof(ADataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, b_elems * sizeof(BDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&E_dev, e_elems * sizeof(EDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, a_elems * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, b_elems * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(E_dev, 0, e_elems * sizeof(EDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // D tensors: shape == E ([G..,M..,N..]), row-major packed strides == E_strides.
    std::array<const void*, kNumD> ds_ptr{};
    std::array<std::vector<ck_tile::index_t>, kNumD> Ds_dims{};
    std::array<std::vector<ck_tile::index_t>, kNumD> Ds_strides{};
    for(int i = 0; i < kNumD; ++i)
    {
        if(hipMalloc(&D_dev[i], e_elems * sizeof(DBaseDataType)) != hipSuccess)
        {
            cleanup();
            return -1;
        }
        if(hipMemcpy(D_dev[i], d_ptrs[i], e_elems * sizeof(DBaseDataType), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
        ds_ptr[i]     = D_dev[i];
        Ds_dims[i]    = E_dims;
        Ds_strides[i] = E_strides;
    }

    ck_tile::BatchedContractionHostArgs<kNumD> args(
        /*a_ptr*/ A_dev,
        /*b_ptr*/ B_dev,
        /*ds_ptr*/ ds_ptr,
        /*e_ptr*/ E_dev,
        /*k_batch*/ static_cast<ck_tile::index_t>(k_batch),
        /*A_dims*/ A_dims,
        /*B_dims*/ B_dims,
        /*Ds_dims*/ Ds_dims,
        /*E_dims*/ E_dims,
        /*A_strides*/ A_strides,
        /*B_strides*/ B_strides,
        /*Ds_strides*/ Ds_strides,
        /*E_strides*/ E_strides);

    const bool do_time = (time_ms != nullptr);
    // Defaults mirror the Old-TE profiler so bridge-vs-Old-TE timings are
    // apples-to-apples out of the box: warmup=50, repeat=100, GPU timer,
    // flush_cache=true, rotating_count=1000. warmup/repeat remain env-overridable
    // (CK_TILE_BENCH_WARMUP / CK_TILE_BENCH_REPEAT) for custom sweeps -- set the
    // same values on both sides to keep the comparison fair.
    const int warmup = do_time ? env_int("CK_TILE_BENCH_WARMUP", 50) : 0;
    const int repeat = do_time ? env_int("CK_TILE_BENCH_REPEAT", 100) : 1;
    // stream_config field order: stream_id, time_kernel, log_level, cold_niters
    // (warmup), nrepeat, is_gpu_timer, flush_cache, rotating_count.
    ck_tile::stream_config stream_cfg{
        nullptr,           // stream_id_
        do_time,           // time_kernel_
        0,                 // log_level_
        warmup,            // cold_niters_
        repeat,            // nrepeat_
        do_time,           // is_gpu_timer_
        do_time,           // flush_cache_
        do_time ? 1000 : 1 // rotating_count_
    };

    float exec_time = 0.0f;
    try
    {
        exec_time = SelectedKernel::launch(args, stream_cfg);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_batched_contraction: launch threw: " << e.what() << "\n";
        cleanup();
        return -1;
    }
    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_batched_contraction: kernel reports unsupported args\n";
        cleanup();
        return -2;
    }

    if(hipMemcpy(E_host, E_dev, e_elems * sizeof(EDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

} // extern "C"
