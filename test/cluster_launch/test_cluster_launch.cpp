// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"

using ::ck::DeviceMem;

constexpr int kBlockSize = 32;

//
// Test kernels for cluster launch via ck::launch_and_time_kernel with cluster_dim.
//

// Trivial kernel: each thread writes threadIdx.x + blockIdx.x * blockDim.x to output.
__global__ void basic_cluster_kernel(float* __restrict__ out, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n)
    {
        out[tid] = static_cast<float>(tid);
    }
}

// Kernel: each thread writes its cluster_id_x to output.
__global__ void cluster_builtin_kernel(int* __restrict__ out, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n)
        return;

    out[tid] = __builtin_amdgcn_cluster_id_x();
}

// Kernel: uses dynamic LDS under cluster launch.
// Each thread writes threadIdx.x to LDS, syncs, then reads it back to output.
extern __shared__ float lds_buffer[];

__global__ void cluster_lds_kernel(float* __restrict__ out, int n)
{
    int tid      = threadIdx.x;
    int block_id = blockIdx.x;

    if(tid >= n)
        return;

    lds_buffer[tid] = static_cast<float>(tid + block_id * 1000);
    __syncthreads();

    out[block_id * n + tid] = lds_buffer[tid];
}

TEST(ClusterLaunch, BasicKernel)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int numBlocks = 2;
    constexpr int N         = kBlockSize * numBlocks;

    DeviceMem out_mem(N * sizeof(float));
    out_mem.SetZero();

    StreamConfig stream_config;
    stream_config.time_kernel_ = false;

    dim3 cluster_dim(numBlocks, 1, 1);
    dim3 grid_dim(numBlocks);
    dim3 block_dim(kBlockSize);

    ck::launch_and_time_kernel(stream_config,
                               basic_cluster_kernel,
                               grid_dim,
                               cluster_dim,
                               block_dim,
                               std::size_t{0},
                               static_cast<float*>(out_mem.GetDeviceBuffer()),
                               N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<float> out_host(N);
    out_mem.FromDevice(out_host.data());

    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(static_cast<float>(i), out_host[i]) << "Mismatch at index " << i;
    }
}

TEST(ClusterLaunch, ClusterBuiltins)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    // Use 4 blocks with cluster_dim=2 to get 2 clusters:
    //   Cluster 0: blocks 0, 1
    //   Cluster 1: blocks 2, 3
    constexpr int clusterSize = 2;
    constexpr int numBlocks   = 4;
    constexpr int N           = kBlockSize * numBlocks;

    DeviceMem out_mem(N * sizeof(int));
    out_mem.SetZero();

    StreamConfig stream_config;
    stream_config.time_kernel_ = false;

    dim3 cluster_dim(clusterSize, 1, 1);
    dim3 grid_dim(numBlocks);
    dim3 block_dim(kBlockSize);

    ck::launch_and_time_kernel(stream_config,
                               cluster_builtin_kernel,
                               grid_dim,
                               cluster_dim,
                               block_dim,
                               std::size_t{0},
                               static_cast<int*>(out_mem.GetDeviceBuffer()),
                               N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<int> out_host(N);
    out_mem.FromDevice(out_host.data());

    // cluster_id_x = blockIdx.x / clusterSize
    // Blocks 0,1 -> cluster 0; Blocks 2,3 -> cluster 1
    for(int block = 0; block < numBlocks; ++block)
    {
        int expected_cluster_id = block / clusterSize;
        for(int t = 0; t < kBlockSize; ++t)
        {
            int idx = block * kBlockSize + t;
            EXPECT_EQ(expected_cluster_id, out_host[idx])
                << "Block " << block << ", thread " << t << " reported wrong cluster_id_x";
        }
    }
}

TEST(ClusterLaunch, WithLDS)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This test is not supported on asicRevision=0";
    }

    constexpr int numBlocks = 2;
    constexpr int N         = kBlockSize;

    DeviceMem out_mem(N * numBlocks * sizeof(float));
    out_mem.SetZero();

    StreamConfig stream_config;
    stream_config.time_kernel_ = false;

    dim3 cluster_dim(numBlocks, 1, 1);
    dim3 grid_dim(numBlocks);
    dim3 block_dim(N);
    std::size_t lds_bytes = N * sizeof(float);

    ck::launch_and_time_kernel(stream_config,
                               cluster_lds_kernel,
                               grid_dim,
                               cluster_dim,
                               block_dim,
                               lds_bytes,
                               static_cast<float*>(out_mem.GetDeviceBuffer()),
                               N);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<float> out_host(N * numBlocks);
    out_mem.FromDevice(out_host.data());

    for(int block = 0; block < numBlocks; ++block)
    {
        for(int t = 0; t < N; ++t)
        {
            float expected = static_cast<float>(t + block * 1000);
            EXPECT_EQ(expected, out_host[block * N + t])
                << "Block " << block << ", thread " << t << " LDS mismatch";
        }
    }
}
