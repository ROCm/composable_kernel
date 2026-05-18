// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/amd_cluster_load.hpp"

#include <cstring>

using ::ck::DeviceMem;

// Number of elements per WGP (Wave32)
constexpr int kTileSize = 32;

//
// cluster_load_async: Global → LDS with a WGP participation mask.
// Templated kernels covering 1-byte (char), 4-byte (int), 8-byte (int2),
// and 16-byte (int4) async loads.
//

// Shared memory declared as raw bytes; kernels cast as needed.
extern __shared__ char shared_lds[];

// --- Templated kernels ----------------------------------------------------

// Single WGP, async load global → LDS, copy LDS → output. mask = 0x1.
template <typename T>
__global__ void
cluster_load_async_single_wgp_kernel(const T* __restrict__ in, T* __restrict__ out, int n)
{
    int tid = threadIdx.x;

    T* lds = reinterpret_cast<T*>(shared_lds);

    if(tid < n)
    {
        auto* lds_ptr = reinterpret_cast<__attribute__((address_space(3))) void*>(
            reinterpret_cast<uintptr_t>(&lds[tid]));
        auto* g_ptr = reinterpret_cast<__attribute__((address_space(1))) const void*>(
            reinterpret_cast<uintptr_t>(&in[tid]));

        ck::cluster_load_async<sizeof(T)>(lds_ptr, g_ptr, 0x1);
    }

    ck::cluster_load_async_wait();
    __syncthreads();

    if(tid < n)
    {
        out[tid] = lds[tid];
    }
}

// Multi-WGP broadcast. mask = (1 << numWGPs) - 1.
template <typename T>
__global__ void
cluster_load_async_multi_wgp_kernel(const T* __restrict__ in, T* __restrict__ out, int n, int mask)
{
    int tid      = threadIdx.x;
    int block_id = blockIdx.x;

    T* lds = reinterpret_cast<T*>(shared_lds);

    if(tid < n)
    {
        auto* lds_ptr = reinterpret_cast<__attribute__((address_space(3))) void*>(
            reinterpret_cast<uintptr_t>(&lds[tid]));
        auto* g_ptr = reinterpret_cast<__attribute__((address_space(1))) const void*>(
            reinterpret_cast<uintptr_t>(&in[tid]));

        ck::cluster_load_async<sizeof(T)>(lds_ptr, g_ptr, mask);
    }

    ck::cluster_load_async_wait();
    __syncthreads();

    if(tid < n)
    {
        out[block_id * n + tid] = lds[tid];
    }
}

// Partial mask (non-contiguous WGPs). Exports flat_id for host verification.
template <typename T>
__global__ void cluster_load_async_partial_mask_kernel(
    const T* __restrict__ in, T* __restrict__ out, int n, int mask, int* __restrict__ flat_ids)
{
    int tid      = threadIdx.x;
    int block_id = blockIdx.x;

    int cluster_id     = __builtin_amdgcn_cluster_workgroup_flat_id();
    bool participating = (mask >> cluster_id) & 1;

    if(tid == 0)
        flat_ids[block_id] = cluster_id;

    T* lds = reinterpret_cast<T*>(shared_lds);

    // Initialize LDS to sentinel (all 0xFF bytes)
    if(tid < n)
    {
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(&lds[tid]);
#pragma unroll
        for(int i = 0; i < static_cast<int>(sizeof(T)); ++i)
        {
            byte_ptr[i] = 0xFF;
        }
    }
    __syncthreads();

    if(tid < n && participating)
    {
        auto* lds_ptr = reinterpret_cast<__attribute__((address_space(3))) void*>(
            reinterpret_cast<uintptr_t>(&lds[tid]));
        auto* g_ptr = reinterpret_cast<__attribute__((address_space(1))) const void*>(
            reinterpret_cast<uintptr_t>(&in[tid]));

        ck::cluster_load_async<sizeof(T)>(lds_ptr, g_ptr, mask);
        ck::cluster_load_async_wait();
    }
    __syncthreads();

    if(tid < n)
    {
        out[block_id * n + tid] = lds[tid];
    }
}

// LDS bounds check — sentinel region adjacent to loaded tile remains zero.
template <typename T>
__global__ void
cluster_load_async_bounds_check_kernel(const T* __restrict__ in, T* __restrict__ out, int n)
{
    int tid = threadIdx.x;
    T* lds  = reinterpret_cast<T*>(shared_lds);

    // LDS layout: [tile of n elements] [sentinel region of n elements]
    {
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(&lds[tid]);
        unsigned char* sent_ptr = reinterpret_cast<unsigned char*>(&lds[tid + n]);
#pragma unroll
        for(int i = 0; i < static_cast<int>(sizeof(T)); ++i)
        {
            byte_ptr[i] = 0;
            sent_ptr[i] = 0;
        }
    }
    __syncthreads();

    if(tid < n)
    {
        auto* lds_ptr = reinterpret_cast<__attribute__((address_space(3))) void*>(
            reinterpret_cast<uintptr_t>(&lds[tid]));
        auto* g_ptr = reinterpret_cast<__attribute__((address_space(1))) const void*>(
            reinterpret_cast<uintptr_t>(&in[tid]));

        ck::cluster_load_async<sizeof(T)>(lds_ptr, g_ptr, 0x1);
        ck::cluster_load_async_wait();
    }
    __syncthreads();

    out[tid]     = lds[tid];
    out[tid + n] = lds[tid + n];
}

// --- Fill helpers ---------------------------------------------------------

template <typename T>
void fill_src(std::vector<T>& src, int base);

template <>
void fill_src<char>(std::vector<char>& src, int base)
{
    for(int i = 0; i < static_cast<int>(src.size()); ++i)
        src[i] = static_cast<char>((base + i) & 0x7F);
}

template <>
void fill_src<int>(std::vector<int>& src, int base)
{
    for(int i = 0; i < static_cast<int>(src.size()); ++i)
        src[i] = base + i;
}

template <>
void fill_src<int2>(std::vector<int2>& src, int base)
{
    for(int i = 0; i < static_cast<int>(src.size()); ++i)
        src[i] = {base + i, base + 100 + i};
}

template <>
void fill_src<int4>(std::vector<int4>& src, int base)
{
    for(int i = 0; i < static_cast<int>(src.size()); ++i)
        src[i] = {base + i, base + 100 + i, base + 200 + i, base + 300 + i};
}

// --- GTest typed test suite -----------------------------------------------

template <typename T>
class ClusterLoadAsyncTyped : public ::testing::Test
{
};

using ClusterLoadAsyncTypes = ::testing::Types<char, int, int2, int4>;
TYPED_TEST_SUITE(ClusterLoadAsyncTyped, ClusterLoadAsyncTypes);

TYPED_TEST(ClusterLoadAsyncTyped, SingleWGP_AsyncToLDS)
{
    using T = TypeParam;

    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int N = kTileSize;

    DeviceMem in_mem(N * sizeof(T));
    DeviceMem out_mem(N * sizeof(T));

    std::vector<T> in_host(N);
    fill_src<T>(in_host, 0);
    in_mem.ToDevice(in_host.data());
    out_mem.SetZero();

    dim3 grid(1);
    dim3 block(N);
    std::size_t lds_bytes = N * sizeof(T);

    cluster_load_async_single_wgp_kernel<T>
        <<<grid, block, lds_bytes>>>(static_cast<const T*>(in_mem.GetDeviceBuffer()),
                                     static_cast<T*>(out_mem.GetDeviceBuffer()),
                                     N);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<T> out_host(N);
    out_mem.FromDevice(out_host.data());

    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(std::memcmp(&in_host[i], &out_host[i], sizeof(T)), 0)
            << "Mismatch at index " << i;
    }
}

TYPED_TEST(ClusterLoadAsyncTyped, MultiWGP_AsyncBroadcastToLDS)
{
    using T = TypeParam;

    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int N       = kTileSize;
    constexpr int numWGPs = 2;
    constexpr int mask    = (1 << numWGPs) - 1; // 0x3

    DeviceMem in_mem(N * sizeof(T));
    DeviceMem out_mem(N * numWGPs * sizeof(T));

    std::vector<T> in_host(N);
    fill_src<T>(in_host, 42);
    in_mem.ToDevice(in_host.data());
    out_mem.SetZero();

    dim3 grid(numWGPs);
    dim3 block(N);
    std::size_t lds_bytes = N * sizeof(T);

    ck::launch_and_time_kernel(StreamConfig{},
                               cluster_load_async_multi_wgp_kernel<T>,
                               grid,
                               dim3(numWGPs, 1, 1),
                               block,
                               lds_bytes,
                               static_cast<const T*>(in_mem.GetDeviceBuffer()),
                               static_cast<T*>(out_mem.GetDeviceBuffer()),
                               N,
                               mask);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<T> out_host(N * numWGPs);
    out_mem.FromDevice(out_host.data());

    for(int wgp = 0; wgp < numWGPs; ++wgp)
    {
        for(int i = 0; i < N; ++i)
        {
            EXPECT_EQ(std::memcmp(&in_host[i], &out_host[wgp * N + i], sizeof(T)), 0)
                << "Mismatch at WGP " << wgp << ", index " << i;
        }
    }
}

TYPED_TEST(ClusterLoadAsyncTyped, PartialMask_AsyncNonContiguous)
{
    using T = TypeParam;

    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int N           = kTileSize;
    constexpr int clusterSize = 4;
    constexpr int mask        = 0x5; // WGP 0 and WGP 2

    DeviceMem in_mem(N * sizeof(T));
    DeviceMem out_mem(N * clusterSize * sizeof(T));
    DeviceMem flat_id_mem(clusterSize * sizeof(int));

    std::vector<T> in_host(N);
    fill_src<T>(in_host, 50);
    in_mem.ToDevice(in_host.data());
    out_mem.SetZero();
    flat_id_mem.SetZero();

    dim3 grid(clusterSize);
    dim3 block(N);
    std::size_t lds_bytes = N * sizeof(T);

    ck::launch_and_time_kernel(StreamConfig{},
                               cluster_load_async_partial_mask_kernel<T>,
                               grid,
                               dim3(clusterSize, 1, 1),
                               block,
                               lds_bytes,
                               static_cast<const T*>(in_mem.GetDeviceBuffer()),
                               static_cast<T*>(out_mem.GetDeviceBuffer()),
                               N,
                               mask,
                               static_cast<int*>(flat_id_mem.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<T> out_host(N * clusterSize);
    out_mem.FromDevice(out_host.data());

    std::vector<int> flat_ids(clusterSize);
    flat_id_mem.FromDevice(flat_ids.data());

    T sentinel;
    std::memset(&sentinel, 0xFF, sizeof(T));

    for(int wgp = 0; wgp < clusterSize; ++wgp)
    {
        bool participating = (mask >> flat_ids[wgp]) & 1;
        for(int i = 0; i < N; ++i)
        {
            if(participating)
            {
                EXPECT_EQ(std::memcmp(&in_host[i], &out_host[wgp * N + i], sizeof(T)), 0)
                    << "Participating WGP " << wgp << " (flat_id=" << flat_ids[wgp]
                    << ") mismatch at index " << i;
            }
            else
            {
                EXPECT_EQ(std::memcmp(&sentinel, &out_host[wgp * N + i], sizeof(T)), 0)
                    << "Non-participating WGP " << wgp << " (flat_id=" << flat_ids[wgp]
                    << ") should have sentinel at index " << i;
            }
        }
    }
}

TYPED_TEST(ClusterLoadAsyncTyped, LDS_BoundsCheck)
{
    using T = TypeParam;

    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int N = kTileSize;

    DeviceMem in_mem(N * sizeof(T));
    DeviceMem out_mem(2 * N * sizeof(T));

    std::vector<T> in_host(N);
    fill_src<T>(in_host, 1);
    in_mem.ToDevice(in_host.data());
    out_mem.SetZero();

    dim3 grid(1);
    dim3 block(N);
    std::size_t lds_bytes = 2 * N * sizeof(T);

    cluster_load_async_bounds_check_kernel<T>
        <<<grid, block, lds_bytes>>>(static_cast<const T*>(in_mem.GetDeviceBuffer()),
                                     static_cast<T*>(out_mem.GetDeviceBuffer()),
                                     N);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<T> out_host(2 * N);
    out_mem.FromDevice(out_host.data());

    // Tile region should match input
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(std::memcmp(&in_host[i], &out_host[i], sizeof(T)), 0)
            << "Tile mismatch at index " << i;
    }

    // Sentinel region should remain zero
    T zero;
    std::memset(&zero, 0, sizeof(T));
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(std::memcmp(&zero, &out_host[N + i], sizeof(T)), 0)
            << "Sentinel corrupted at index " << i;
    }
}
