// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck_tile/host/device_prop.hpp"
#include <cstring>
#include <vector>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

static constexpr int NUM_LANES = 32; // Wave32

// Single-WGP kernel: each lane loads from src[lane_id] using cluster_multicast_load.
template <typename T>
struct ClusterLoadKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    CK_TILE_DEVICE void operator()(const T* __restrict__ src, T* __restrict__ dst, int mask) const
    {
        int lane_id = threadIdx.x;
        T result    = ck_tile::cluster_multicast_load(src + lane_id, mask);
        ck_tile::s_waitcnt<0>();
        dst[lane_id] = result;
    }
};

// Single-WGP test helper: 1 WGP, 32 threads, per-lane addressed load.
template <typename T>
void run_single_wgp_test(const std::vector<T>& h_src, int mask, const char* test_name)
{
    std::vector<T> h_dst(NUM_LANES);

    ck_tile::DeviceMem d_src(NUM_LANES * sizeof(T));
    ck_tile::DeviceMem d_dst(NUM_LANES * sizeof(T));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(ClusterLoadKernel<T>{},
                                       dim3(1),
                                       dim3(NUM_LANES),
                                       0,
                                       static_cast<const T*>(d_src.GetDeviceBuffer()),
                                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                                       mask);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    for(int i = 0; i < NUM_LANES; i++)
        EXPECT_EQ(std::memcmp(&h_dst[i], &h_src[i], sizeof(T)), 0)
            << test_name << " mismatch at lane " << i;
}

// --- Group 1: Bit-width correctness (B32, B64, B128), single WGP, mask=0x1 ---

TEST(SingleWGP, B32_AllLanes)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 100 + i;
    run_single_wgp_test<int>(src, 0x1, "B32_AllLanes");
}

TEST(SingleWGP, B64_AllLanes)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    std::vector<int2> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {100 + i, 200 + i};
    run_single_wgp_test<int2>(src, 0x1, "B64_AllLanes");
}

TEST(SingleWGP, B128_AllLanes)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    std::vector<int4> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {100 + i, 200 + i, 300 + i, 400 + i};
    run_single_wgp_test<int4>(src, 0x1, "B128_AllLanes");
}

// --- Group 2: M0 mask semantics (single WGP, varying mask) ---
// Only masks where this WGP is the sole participant are safe with 1 WGP.

TEST(M0Mask, ZeroMask_NonMulticast) // mask=0x0: non-multicast path
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 100 + i;
    run_single_wgp_test<int>(src, 0x0, "ZeroMask_NonMulticast");
}

TEST(M0Mask, SingleBit_WGP0) // mask=0x1: only WGP 0 participates
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 200 + i;
    run_single_wgp_test<int>(src, 0x1, "SingleBit_WGP0");
}

// --- Group 3: Multi-WGP broadcast ---
// All WGPs in a cluster load from the same address. Launched with the cluster
// dim overload of make_kernel so WGPs are co-located on the same SE with
// sequential flat IDs 0..N-1.

template <typename T>
struct MulticastBroadcastKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    CK_TILE_DEVICE void operator()(const T* __restrict__ shared_src,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids,
                                   int num_wgs) const
    {
        int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();

        if(threadIdx.x == 0)
            diag_ids[blockIdx.x] = flat_id;

        int mask = (1 << num_wgs) - 1;
        T result = ck_tile::cluster_multicast_load(shared_src, mask);
        ck_tile::s_waitcnt<0>();

        dst[blockIdx.x * blockDim.x + threadIdx.x] = result;
    }
};

// Broadcast test helper: launches num_wgs WGPs as a cluster, all loading from same address.
// Assumption: cluster launch guarantees blockIdx.x == flat_id (verified by diagnostic check).
// If this assumption breaks on future hardware, the flat_id check will fail and alert us.
template <typename T>
void run_broadcast_test(int num_wgs, const T& src_val, const char* test_name)
{
    const int total_threads = num_wgs * NUM_LANES;

    std::vector<T> h_dst(total_threads);
    std::vector<int> h_diag_ids(num_wgs);

    ck_tile::DeviceMem d_src(sizeof(T));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(T));
    ck_tile::DeviceMem d_diag_ids(num_wgs * sizeof(int));
    d_src.ToDevice(&src_val);
    d_dst.SetBytePattern(0xFF);
    d_diag_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(MulticastBroadcastKernel<T>{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const T*>(d_src.GetDeviceBuffer()),
                                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_diag_ids.GetDeviceBuffer()),
                                       num_wgs);
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_diag_ids.FromDevice(h_diag_ids.data());

    printf("  %s: flat IDs = {", test_name);
    for(int i = 0; i < num_wgs; i++)
        printf("%s%d", i ? ", " : "", h_diag_ids[i]);
    printf("}\n");

    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << "blockIdx.x=" << i << " expected flat_id=" << i;

    for(int i = 0; i < total_threads; i++)
        EXPECT_EQ(std::memcmp(&h_dst[i], &src_val, sizeof(T)), 0)
            << "Broadcast mismatch at thread " << i;
}

TEST(MultiWGP, Broadcast_2WGP_B32)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    run_broadcast_test<int>(2, static_cast<int>(0x13579BDF), "Broadcast_2WGP_B32");
}

TEST(MultiWGP, Broadcast_4WGP_B32)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    run_broadcast_test<int>(4, static_cast<int>(0x13579BDF), "Broadcast_4WGP_B32");
}

TEST(MultiWGP, Broadcast_5WGP_B32)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    run_broadcast_test<int>(5, static_cast<int>(0x13579BDF), "Broadcast_5WGP_B32");
}

TEST(MultiWGP, Broadcast_2WGP_B64)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    int2 src = {static_cast<int>(0x13579BDF), static_cast<int>(0x2468ACE0)};
    run_broadcast_test<int2>(2, src, "Broadcast_2WGP_B64");
}

TEST(MultiWGP, Broadcast_4WGP_B64)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    int2 src = {static_cast<int>(0x13579BDF), static_cast<int>(0x2468ACE0)};
    run_broadcast_test<int2>(4, src, "Broadcast_4WGP_B64");
}

TEST(MultiWGP, Broadcast_4WGP_B128)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    int4 src = {static_cast<int>(0x13579BDF),
                static_cast<int>(0x2468ACE0),
                0x12345678,
                static_cast<int>(0x76543210)};
    run_broadcast_test<int4>(4, src, "Broadcast_4WGP_B128");
}

TEST(MultiWGP, Broadcast_6WGP_B128)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    int4 src = {static_cast<int>(0x13579BDF),
                static_cast<int>(0x2468ACE0),
                0x12345678,
                static_cast<int>(0x76543210)};
    run_broadcast_test<int4>(6, src, "Broadcast_6WGP_B128");
}

// --- Group 4: Partial broadcast (subset of WGPs participate) ---
// Non-contiguous mask: only WGPs whose bit is set issue cluster_multicast_load,
// the rest use a regular global load to avoid deadlock.

template <typename T>
struct PartialBroadcastKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    CK_TILE_DEVICE void operator()(const T* __restrict__ shared_src,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids,
                                   int mask) const
    {
        int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();

        if(threadIdx.x == 0)
            diag_ids[blockIdx.x] = flat_id;

        T result;
        if((mask >> flat_id) & 1)
        {
            result = ck_tile::cluster_multicast_load(shared_src, mask);
            ck_tile::s_waitcnt<0>();
        }
        else
        {
            result = *shared_src;
        }

        dst[blockIdx.x * blockDim.x + threadIdx.x] = result;
    }
};

TEST(PartialBroadcast, NonContiguous_4WGP_Mask0x5) // mask=0x5: WGPs 0 & 2
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    const int num_wgs       = 4;
    const int mask          = 0x5; // binary 0101
    const int total_threads = num_wgs * NUM_LANES;
    const int src_val       = static_cast<int>(0x13579BDF);

    std::vector<int> h_dst(total_threads);
    std::vector<int> h_diag_ids(num_wgs);

    ck_tile::DeviceMem d_src(sizeof(int));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(int));
    ck_tile::DeviceMem d_diag_ids(num_wgs * sizeof(int));
    d_src.ToDevice(&src_val);
    d_dst.SetBytePattern(0xFF);
    d_diag_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(PartialBroadcastKernel<int>{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_diag_ids.GetDeviceBuffer()),
                                       mask);
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_diag_ids.FromDevice(h_diag_ids.data());

    printf("  PartialBroadcast: flat IDs = {");
    for(int i = 0; i < num_wgs; i++)
        printf("%s%d", i ? ", " : "", h_diag_ids[i]);
    printf("}, mask=0x%X\n", static_cast<unsigned>(mask));

    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << "blockIdx.x=" << i << " expected flat_id=" << i;

    for(int i = 0; i < total_threads; i++)
        EXPECT_EQ(std::memcmp(&h_dst[i], &src_val, sizeof(int)), 0) << "Mismatch at thread " << i;
}

// --- Group 5: Concurrent multicast groups ---
// Two independent broadcast groups within the same cluster, each with its own
// mask and source address. Verifies no cross-talk between concurrent broadcasts.

template <typename T>
struct ConcurrentGroupsKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    CK_TILE_DEVICE void operator()(const T* __restrict__ src_a,
                                   const T* __restrict__ src_b,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids) const
    {
        int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();

        if(threadIdx.x == 0)
            diag_ids[blockIdx.x] = flat_id;

        T result;
        if(flat_id < 2)
        {
            result = ck_tile::cluster_multicast_load(src_a, 0x3); // WGPs 0&1
            ck_tile::s_waitcnt<0>();
        }
        else
        {
            result = ck_tile::cluster_multicast_load(src_b, 0xC); // WGPs 2&3
            ck_tile::s_waitcnt<0>();
        }

        dst[blockIdx.x * blockDim.x + threadIdx.x] = result;
    }
};

// Concurrent groups test helper: 4 WGPs, two independent broadcast groups.
// Assumption: cluster launch guarantees blockIdx.x == flat_id (verified by diagnostic check).
template <typename T>
void run_concurrent_groups_test(const T& val_a, const T& val_b, const char* test_name)
{
    const int num_wgs       = 4;
    const int total_threads = num_wgs * NUM_LANES;

    std::vector<T> h_dst(total_threads);
    std::vector<int> h_diag_ids(num_wgs);

    ck_tile::DeviceMem d_src_a(sizeof(T));
    ck_tile::DeviceMem d_src_b(sizeof(T));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(T));
    ck_tile::DeviceMem d_diag_ids(num_wgs * sizeof(int));
    d_src_a.ToDevice(&val_a);
    d_src_b.ToDevice(&val_b);
    d_dst.SetBytePattern(0xFF);
    d_diag_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(ConcurrentGroupsKernel<T>{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const T*>(d_src_a.GetDeviceBuffer()),
                                       static_cast<const T*>(d_src_b.GetDeviceBuffer()),
                                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_diag_ids.GetDeviceBuffer()));
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_diag_ids.FromDevice(h_diag_ids.data());

    printf("  %s: flat IDs = {", test_name);
    for(int i = 0; i < num_wgs; i++)
        printf("%s%d", i ? ", " : "", h_diag_ids[i]);
    printf("}\n");

    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << "blockIdx.x=" << i << " expected flat_id=" << i;

    // WGPs 0&1 should have val_a, WGPs 2&3 should have val_b
    for(int wg = 0; wg < num_wgs; wg++)
    {
        const T& expected = (wg < 2) ? val_a : val_b;
        for(int lane = 0; lane < NUM_LANES; lane++)
        {
            int idx = wg * NUM_LANES + lane;
            EXPECT_EQ(std::memcmp(&h_dst[idx], &expected, sizeof(T)), 0)
                << "WGP " << wg << " lane " << lane << " mismatch";
        }
    }
}

TEST(ConcurrentGroups, TwoGroups_4WGP_B32)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    run_concurrent_groups_test<int>(
        static_cast<int>(0x13579BDF), static_cast<int>(0x2468ACE0), "TwoGroups_4WGP_B32");
}

TEST(ConcurrentGroups, TwoGroups_4WGP_B64)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    int2 val_a = {static_cast<int>(0x13579BDF), static_cast<int>(0x11111111)};
    int2 val_b = {static_cast<int>(0x2468ACE0), static_cast<int>(0x22222222)};
    run_concurrent_groups_test<int2>(val_a, val_b, "TwoGroups_4WGP_B64");
}

// --- Group 6: M0[16] early timeout ---
// M0[16] prevents deadlock when masked WGPs don't all participate.

TEST(EarlyTimeout, SingleWGP_TimeoutBit)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    // mask=0x3 says 2 WGPs but only 1 launched; M0[16] prevents deadlock
    const int mask = 0x3 | (1 << 16);

    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 300 + i;
    run_single_wgp_test<int>(src, mask, "EarlyTimeout");
}

template <typename T>
struct BroadcastWithMaskKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    CK_TILE_DEVICE void operator()(const T* __restrict__ shared_src,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids,
                                   int mask) const
    {
        int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();

        if(threadIdx.x == 0)
            diag_ids[blockIdx.x] = flat_id;

        T result = ck_tile::cluster_multicast_load(shared_src, mask);
        ck_tile::s_waitcnt<0>();

        dst[blockIdx.x * blockDim.x + threadIdx.x] = result;
    }
};

TEST(EarlyTimeout, MultiWGP_TimeoutBit)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load is not supported on asicRevision=0";
    }
    // 2 WGPs launched, mask=0xF claims 4; M0[16] prevents deadlock waiting for WGPs 2&3
    const int num_wgs       = 2;
    const int mask          = 0xF | (1 << 16);
    const int total_threads = num_wgs * NUM_LANES;
    const int src_val       = static_cast<int>(0x13579BDF);

    std::vector<int> h_dst(total_threads);
    std::vector<int> h_diag_ids(num_wgs);

    ck_tile::DeviceMem d_src(sizeof(int));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(int));
    ck_tile::DeviceMem d_diag_ids(num_wgs * sizeof(int));
    d_src.ToDevice(&src_val);
    d_dst.SetBytePattern(0xFF);
    d_diag_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(BroadcastWithMaskKernel<int>{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_diag_ids.GetDeviceBuffer()),
                                       mask);
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_diag_ids.FromDevice(h_diag_ids.data());

    printf("  EarlyTimeout_MultiWGP: flat IDs = {");
    for(int i = 0; i < num_wgs; i++)
        printf("%s%d", i ? ", " : "", h_diag_ids[i]);
    printf("}\n");

    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << "blockIdx.x=" << i << " expected flat_id=" << i;

    for(int i = 0; i < total_threads; i++)
        EXPECT_EQ(std::memcmp(&h_dst[i], &src_val, sizeof(int)), 0) << "Mismatch at thread " << i;
}
