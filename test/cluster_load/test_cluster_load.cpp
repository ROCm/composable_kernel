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
// cluster_load: Global -> VGPRs with a WGP participation mask.
// Templated kernels covering 4-byte (int), 8-byte (int2), and 16-byte (int4) loads.
//

// Helper: fill host vector with deterministic per-component values.
template <typename T>
void fill_src(std::vector<T>& src, int base);

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

// --- Templated kernels ----------------------------------------------------

template <typename T>
__global__ void cluster_load_single_wgp_kernel(const T* __restrict__ in, T* __restrict__ out, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n)
        return;

    T val    = ck::cluster_multicast_load<T>(&in[tid], 0x1);
    out[tid] = val;
}

template <typename T>
__global__ void
cluster_load_multi_wgp_kernel(const T* __restrict__ in, T* __restrict__ out, int n, int mask)
{
    int tid      = threadIdx.x;
    int block_id = blockIdx.x;
    if(tid >= n)
        return;

    T val                   = ck::cluster_multicast_load<T>(&in[tid], mask);
    out[block_id * n + tid] = val;
}

template <typename T>
__global__ void cluster_load_partial_mask_kernel(
    const T* __restrict__ in, T* __restrict__ out, int n, int mask, int* __restrict__ flat_ids)
{
    int tid      = threadIdx.x;
    int block_id = blockIdx.x;

    int wgp_id         = __builtin_amdgcn_cluster_workgroup_flat_id();
    bool participating = (mask >> wgp_id) & 1;

    if(tid == 0)
        flat_ids[block_id] = wgp_id;

    if(tid >= n)
        return;

    if(participating)
    {
        T val                   = ck::cluster_multicast_load<T>(&in[tid], mask);
        out[block_id * n + tid] = val;
    }
    else
    {
        // Write sentinel: all bytes 0xFF
        T sentinel;
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(&sentinel);
#pragma unroll
        for(int i = 0; i < static_cast<int>(sizeof(T)); ++i)
        {
            byte_ptr[i] = 0xFF;
        }
        out[block_id * n + tid] = sentinel;
    }
}

// --- GTest typed test suite -----------------------------------------------

template <typename T>
class ClusterLoadTyped : public ::testing::Test
{
};

using ClusterLoadTypes = ::testing::Types<int, int2, int4>;
TYPED_TEST_SUITE(ClusterLoadTyped, ClusterLoadTypes);

TYPED_TEST(ClusterLoadTyped, SingleWGP_CorrectValues)
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

    ck::launch_and_time_kernel(StreamConfig{},
                               cluster_load_single_wgp_kernel<T>,
                               grid,
                               dim3(1, 1, 1),
                               block,
                               std::size_t{0},
                               static_cast<const T*>(in_mem.GetDeviceBuffer()),
                               static_cast<T*>(out_mem.GetDeviceBuffer()),
                               N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<T> out_host(N);
    out_mem.FromDevice(out_host.data());

    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(std::memcmp(&in_host[i], &out_host[i], sizeof(T)), 0)
            << "Mismatch at index " << i;
    }
}

TYPED_TEST(ClusterLoadTyped, MultiWGP_Broadcast)
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

    ck::launch_and_time_kernel(StreamConfig{},
                               cluster_load_multi_wgp_kernel<T>,
                               grid,
                               dim3(numWGPs, 1, 1),
                               block,
                               std::size_t{0},
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

TYPED_TEST(ClusterLoadTyped, PartialMask_NonContiguous)
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
    fill_src<T>(in_host, 100);
    in_mem.ToDevice(in_host.data());
    out_mem.SetZero();
    flat_id_mem.SetZero();

    dim3 grid(clusterSize);
    dim3 block(N);

    ck::launch_and_time_kernel(StreamConfig{},
                               cluster_load_partial_mask_kernel<T>,
                               grid,
                               dim3(clusterSize, 1, 1),
                               block,
                               std::size_t{0},
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

    // Sentinel: all bytes 0xFF (matches kernel)
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
