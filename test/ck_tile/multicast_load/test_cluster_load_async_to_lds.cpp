// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test suite for cluster_multicast_load_async_to_lds<T> — the CK Tile wrapper
// around CLUSTER_LOAD_ASYNC_TO_LDS_B* (gfx1250 only).
//
// Complements test_cluster_load_multicast.cpp (CLUSTER_LOAD_B, VGPR destination)
// by testing behaviors unique to the async LDS path:
//
//   Group 1:  SingleWGP baseline — B32/B64/B128, mask=0x1 and mask=0x0
//   Group 2:  LDSVisibility — non-requesting waves read LDS after barrier
//   Group 3:  LDS address layout — per-lane VDST strided addressing
//   Group 4:  MultiWGP broadcast — async LDS delivery at cluster scale (1D and 2D cluster dims)
//   Group 5:  ASYNCcnt ordering — CLUSTER and GLOBAL async loads share one counter
//   Group 6:  PartialBroadcast — non-contiguous mask, mixed instruction types
//   Group 8:  MultiWGP + LDSVisibility — canonical GEMM tile-load pattern
//   Group 10: ConcurrentGroups — LDS routing isolation between independent groups
//   Group 11: BufferViewAsyncGet — cluster_async_get() through buffer_view,
//             including ISA-specified INST_OFFSET behaviour
//
// Synchronization primitives used:
//   s_wait_asynccnt<0>() — wait for all pending async LDS writes to complete.
//     ASYNCcnt decrements only when the LDS write is committed and visible to
//     subsequent DS reads on the same wave.
//   block_sync_lds_direct_load<0>() — s_wait_asynccnt<0> + s_barrier_signal/wait.
//     Used when multiple waves in a WG must synchronize after an async LDS load.

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck_tile/host/device_prop.hpp"
#include <cstring>
#include <vector>

#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

static constexpr int NUM_LANES = 32; // Wave32

// ---------------------------------------------------------------------------
// Group 1: SingleWGP baseline
// ---------------------------------------------------------------------------

// Single-WGP kernel: each lane loads src[lane_id] into LDS[lane_id], waits on
// ASYNCcnt, then copies LDS[lane_id] to dst[lane_id]. Used for B32/B64/B128
// baseline and zero-mask degradation tests.
template <typename T>
struct AsyncLDSKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const T* __restrict__ src, T* __restrict__ dst, int mask) const
    {
#ifdef __gfx1250__
        __shared__ T lds_buf[NUM_LANES];

        const int lane_id = threadIdx.x;

        ck_tile::cluster_multicast_load_async_to_lds(
            src + lane_id, ck_tile::to_lds(lds_buf + lane_id), mask);

        ck_tile::s_wait_asynccnt<0>();

        dst[lane_id] = lds_buf[lane_id];
#else
        (void)src;
        (void)dst;
        (void)mask;
#endif
    }
};

// ---------------------------------------------------------------------------
// Test helper (Groups 1/7)
// ---------------------------------------------------------------------------

template <typename T>
void run_async_lds_test(const std::vector<T>& h_src, int mask, const char* test_name)
{
    std::vector<T> h_dst(NUM_LANES);

    ck_tile::DeviceMem d_src(NUM_LANES * sizeof(T));
    ck_tile::DeviceMem d_dst(NUM_LANES * sizeof(T));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(AsyncLDSKernel<T>{},
                                       dim3(1),
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
            << test_name << ": mismatch at lane " << i;
}

// ---------------------------------------------------------------------------
// Group 1: SingleWGP — B32, B64, B128, mask=0x1
// ---------------------------------------------------------------------------

TEST(AsyncLDS, B32_SingleWGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 1000 + i;
    run_async_lds_test<int>(src, 0x1, "B32_SingleWGP");
}

TEST(AsyncLDS, B64_SingleWGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int2> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {1000 + i, 2000 + i};
    run_async_lds_test<int2>(src, 0x1, "B64_SingleWGP");
}

TEST(AsyncLDS, B128_SingleWGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int4> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {1000 + i, 2000 + i, 3000 + i, 4000 + i};
    run_async_lds_test<int4>(src, 0x1, "B128_SingleWGP");
}

// ---------------------------------------------------------------------------
// Group 2: LDSVisibility — cross-wave LDS sharing after async load
// ---------------------------------------------------------------------------
// 4 waves per WG (128 threads). Wave 0 issues the async cluster load into
// LDS[0..31], then all waves synchronize via block_sync_lds_direct_load
// (which waits ASYNCcnt=0 then does s_barrier_signal/wait).
// Waves 1–3 read from the same LDS buffer after the barrier.
// Verifies the core guarantee: non-requesting waves see correct LDS data.
//
// block_sync_lds_direct_load<0>() is used for all waves:
//   - wave 0: asynccnt may be non-zero; it waits before signaling the barrier
//   - waves 1–3: asynccnt is already 0 (no-op), then they signal and wait
// The barrier ensures LDS writes from wave 0 are visible to all waves before
// any wave reads from LDS.

template <typename T>
struct LDSVisibilityKernel
{
    static constexpr int kBlockSize = 4 * NUM_LANES; // 128 threads = 4 waves

    CK_TILE_DEVICE void operator()(const T* __restrict__ src, T* __restrict__ dst, int mask) const
    {
#ifdef __gfx1250__
        __shared__ T lds_buf[NUM_LANES]; // 32 slots, loaded by wave 0's 32 lanes

        const int thread_id = threadIdx.x;
        const int lane_id   = thread_id % NUM_LANES;
        const int wave_id   = thread_id / NUM_LANES;

        if(wave_id == 0)
        {
            ck_tile::cluster_multicast_load_async_to_lds(
                src + lane_id, ck_tile::to_lds(lds_buf + lane_id), mask);
        }

        // All waves call block_sync_lds_direct_load: it issues s_wait_asynccnt (a
        // no-op for waves 1–3 whose count is already 0), then s_barrier_signal/wait.
        // After this call all waves are past the barrier and LDS is safe to read.
        ck_tile::block_sync_lds_direct_load<0>();

        dst[thread_id] = lds_buf[lane_id];
#else
        (void)src;
        (void)dst;
        (void)mask;
#endif
    }
};

template <typename T>
void run_lds_visibility_test(const std::vector<T>& h_src, int mask, const char* test_name)
{
    constexpr int NUM_THREADS = 4 * NUM_LANES;
    std::vector<T> h_dst(NUM_THREADS);

    ck_tile::DeviceMem d_src(NUM_LANES * sizeof(T));
    ck_tile::DeviceMem d_dst(NUM_THREADS * sizeof(T));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(LDSVisibilityKernel<T>{},
                                       dim3(1),
                                       dim3(1),
                                       dim3(NUM_THREADS),
                                       0,
                                       static_cast<const T*>(d_src.GetDeviceBuffer()),
                                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                                       mask);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    for(int thread = 0; thread < NUM_THREADS; thread++)
    {
        int lane = thread % NUM_LANES;
        EXPECT_EQ(std::memcmp(&h_dst[thread], &h_src[lane], sizeof(T)), 0)
            << test_name << ": wave " << thread / NUM_LANES << " lane " << lane << " mismatch";
    }
}

TEST(LDSVisibility, B32_FourWaves)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 1000 + i;
    run_lds_visibility_test<int>(src, 0x1, "B32_FourWaves");
}

TEST(LDSVisibility, B64_FourWaves)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int2> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {1000 + i, 2000 + i};
    run_lds_visibility_test<int2>(src, 0x1, "B64_FourWaves");
}

TEST(LDSVisibility, B128_FourWaves)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int4> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = {1000 + i, 2000 + i, 3000 + i, 4000 + i};
    run_lds_visibility_test<int4>(src, 0x1, "B128_FourWaves");
}

// ---------------------------------------------------------------------------
// Group 3: LDSAddressLayout — per-lane VDST addressing (strided)
// ---------------------------------------------------------------------------
// CLUSTER_LOAD_ASYNC_TO_LDS_B* supplies the LDS destination address via a
// per-lane VGPR (VDST). Each lane independently specifies where in LDS its
// data lands. Groups 1 and 2 use contiguous stride-1 addressing implicitly;
// this group explicitly tests non-contiguous (strided) addressing.
//
// Each lane writes to lds_buf[lane_id * kStride], leaving kStride-1 unused
// slots between lanes. The strided slots are zero-initialized before the
// async load so that any unwritten slot reads back 0 — which cannot
// collide with src[i] = 1000 + i. If the hardware ignores VDST and
// writes to lds_buf[lane_id] instead, lanes 1..31 read from their strided
// slots and find zeros, causing a FAIL.
//
// s_wait_dscnt<0>() after zero-init drains the synchronous DS writes
// before the async load is issued to the same slots, preventing a
// write-after-write race between the two paths.

struct LDSStridedKernel
{
    static constexpr int kBlockSize = NUM_LANES;
    static constexpr int kStride    = 8; // 8 int slots (32 bytes) between lanes

    CK_TILE_DEVICE void
    operator()(const int* __restrict__ src, int* __restrict__ dst, int mask) const
    {
#ifdef __gfx1250__
        __shared__ int lds_buf[NUM_LANES * LDSStridedKernel::kStride];

        const int lane_id = threadIdx.x;

        // Zero-initialize this lane's strided region.
        for(int s = 0; s < LDSStridedKernel::kStride; s++)
            lds_buf[lane_id * LDSStridedKernel::kStride + s] = 0;

        // Drain the synchronous DS writes before issuing the async load
        // to the same slots, avoiding a write-after-write race.
        ck_tile::s_wait_dscnt<0>();

        // Async load: each lane writes to its strided slot via VDST.
        ck_tile::cluster_multicast_load_async_to_lds(
            src + lane_id, ck_tile::to_lds(&lds_buf[lane_id * LDSStridedKernel::kStride]), mask);

        ck_tile::s_wait_asynccnt<0>();

        // Read back the strided slot; any VDST-ignore bug yields 0 here.
        dst[lane_id] = lds_buf[lane_id * LDSStridedKernel::kStride];
#else
        (void)src;
        (void)dst;
        (void)mask;
#endif
    }
};

void run_strided_lds_test(const std::vector<int>& h_src, int mask, const char* test_name)
{
    std::vector<int> h_dst(NUM_LANES);

    ck_tile::DeviceMem d_src(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_dst(NUM_LANES * sizeof(int));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(LDSStridedKernel{},
                                       dim3(1),
                                       dim3(1),
                                       dim3(NUM_LANES),
                                       0,
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       mask);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    for(int i = 0; i < NUM_LANES; i++)
        EXPECT_EQ(h_dst[i], h_src[i]) << test_name << ": mismatch at lane " << i << " (got "
                                      << h_dst[i] << ", want " << h_src[i] << ")";
}

TEST(LDSAddressLayout, B32_Strided_SingleWGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 1000 + i;
    run_strided_lds_test(src, 0x1, "B32_Strided_SingleWGP");
}

// ---------------------------------------------------------------------------
// Group 4: MultiWGP Broadcast — async LDS delivery at cluster scale
// ---------------------------------------------------------------------------
// All WGPs in a cluster load from the same single source value (true broadcast).
// Each lane within a WGP loads from shared_src → lds_buf[lane_id] via per-lane
// VDST, so every LDS slot in every WGP ends up holding the broadcast value.
//
// Wave 0 of each WGP issues the async cluster load; s_wait_asynccnt<0> ensures
// the LDS write is complete before the result is read back.
//
// The flat_id diagnostic mirrors test_cluster_load_multicast.cpp: it confirms
// blockIdx.x == cluster_workgroup_flat_id(), which must hold for the mask
// calculation (1 << num_wgs) - 1 to assign the correct bit to each WGP.
//
// Two representative cases: 2-WGP B32, 4-WGP B128. The multicast scheduler
// and GL1 merging logic is already exhaustively tested in
// test_cluster_load_multicast.cpp; this group confirms the LDS write leg works
// at cluster scale.

template <typename T>
struct AsyncLDSBroadcastKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const T* __restrict__ shared_src,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids,
                                   int num_wgs) const
    {
#ifdef __gfx1250__
        __shared__ T lds_buf[NUM_LANES];

        const int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();
        const int lane_id = threadIdx.x;

        if(lane_id == 0)
            diag_ids[blockIdx.x] = flat_id;

        const int mask = (1 << num_wgs) - 1;

        // True broadcast: all lanes load from the same address. Each lane's copy
        // lands in its own LDS slot via per-lane VDST (lds_buf + lane_id).
        ck_tile::cluster_multicast_load_async_to_lds(
            shared_src, ck_tile::to_lds(lds_buf + lane_id), mask);

        // Single wave per WGP: no barrier needed, just wait for the async LDS write.
        ck_tile::s_wait_asynccnt<0>();

        dst[blockIdx.x * blockDim.x + lane_id] = lds_buf[lane_id];
#else
        (void)shared_src;
        (void)dst;
        (void)diag_ids;
        (void)num_wgs;
#endif
    }
};

template <typename T>
void run_async_lds_broadcast_test(int num_wgs, const T& src_val, const char* test_name)
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
    auto kernel = ck_tile::make_kernel(AsyncLDSBroadcastKernel<T>{},
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

    // Verify flat IDs are contiguous 0..num_wgs-1 (cluster layout assumption).
    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << test_name << ": blockIdx.x=" << i
                                    << " expected flat_id=" << i << " got " << h_diag_ids[i];

    // Verify every lane in every WGP received the broadcast value.
    for(int i = 0; i < total_threads; i++)
        EXPECT_EQ(std::memcmp(&h_dst[i], &src_val, sizeof(T)), 0)
            << test_name << ": broadcast mismatch at thread " << i;
}

TEST(MultiWGPBroadcast, B32_2WGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    run_async_lds_broadcast_test<int>(2, static_cast<int>(0xDECAFBAD), "B32_2WGP");
}

TEST(MultiWGPBroadcast, B128_4WGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    int4 src_val = {static_cast<int>(0xDECAFBAD),
                    static_cast<int>(0xDEADBEEF),
                    0x12345678,
                    static_cast<int>(0xAAAAAAAA)};
    run_async_lds_broadcast_test<int4>(4, src_val, "B128_4WGP");
}

// 2D cluster broadcast: cluster_dim=dim3(2,2,1), matching the default 2D
// cluster layout used by ck_tile cluster pipelines.
// The flat block index is computed as blockIdx.x + blockIdx.y * gridDim.x.
// mask=0xF covers all 4 WGPs (bits 0-3).
struct AsyncLDS2DClusterKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const int* __restrict__ shared_src,
                                   int* __restrict__ dst,
                                   int* __restrict__ flat_ids) const
    {
#ifdef __gfx1250__
        __shared__ int lds_buf[NUM_LANES];

        const int block_flat_id = blockIdx.x + blockIdx.y * gridDim.x;
        const int lane_id       = threadIdx.x;

        if(lane_id == 0)
            flat_ids[block_flat_id] = __builtin_amdgcn_cluster_workgroup_flat_id();

        constexpr int mask = 0xF; // 4 WGPs in a 2x2 cluster

        ck_tile::cluster_multicast_load_async_to_lds(
            shared_src, ck_tile::to_lds(lds_buf + lane_id), mask);

        ck_tile::s_wait_asynccnt<0>();

        dst[block_flat_id * NUM_LANES + lane_id] = lds_buf[lane_id];
#else
        (void)shared_src;
        (void)dst;
        (void)flat_ids;
#endif
    }
};

TEST(MultiWGPBroadcast, B32_2x2Cluster)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    constexpr int kWGPsX     = 2;
    constexpr int kWGPsY     = 2;
    constexpr int kTotalWGPs = kWGPsX * kWGPsY;

    const int src_val = static_cast<int>(0xC0FFEE42);

    std::vector<int> h_dst(kTotalWGPs * NUM_LANES);
    std::vector<int> h_flat_ids(kTotalWGPs);

    ck_tile::DeviceMem d_src(sizeof(int));
    ck_tile::DeviceMem d_dst(kTotalWGPs * NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_flat_ids(kTotalWGPs * sizeof(int));
    d_src.ToDevice(&src_val);
    d_dst.SetBytePattern(0xFF);
    d_flat_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(AsyncLDS2DClusterKernel{},
                                       dim3(kWGPsX, kWGPsY, 1),
                                       dim3(kWGPsX, kWGPsY, 1),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_flat_ids.GetDeviceBuffer()));
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_flat_ids.FromDevice(h_flat_ids.data());

    // cluster_workgroup_flat_id() uses x-major ordering matching
    // blockIdx.x + blockIdx.y * gridDim.x, so flat_ids[i] must equal i exactly.
    for(int i = 0; i < kTotalWGPs; i++)
        EXPECT_EQ(h_flat_ids[i], i) << "B32_2x2Cluster: flat_id mismatch at block " << i;

    // Every lane in every WGP must hold the broadcast value.
    for(int i = 0; i < kTotalWGPs * NUM_LANES; i++)
        EXPECT_EQ(h_dst[i], src_val) << "B32_2x2Cluster: mismatch at index " << i;
}

// ---------------------------------------------------------------------------
// Group 7: ZeroMask — mask=0x0 degrades to non-multicast async load
// ---------------------------------------------------------------------------
// ISA spec: "If M0[15:0] == 0, this is treated as a non-Cluster-multicast load:
// return only to the requesting WGP (it is not treated as 'do not return to
// any wave')." Data still lands in the requesting WGP's LDS — no deadlock,
// no lost load.

TEST(AsyncLDS, B32_ZeroMask)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        src[i] = 500 + i;
    run_async_lds_test<int>(src, 0x0, "B32_ZeroMask");
}

// ---------------------------------------------------------------------------
// Group 5: ASYNCcnt Ordering — CLUSTER_LOAD_ASYNC_TO_LDS + GLOBAL_LOAD_ASYNC_TO_LDS
// ---------------------------------------------------------------------------
// Both instructions share a single ASYNCcnt on gfx1250, so one
// s_wait_asynccnt<0>() is sufficient to guarantee both async LDS writes
// are complete.
//
// A single wave issues both instructions back-to-back with no wait between:
//   1. cluster_multicast_load_async_to_lds(src_a + lane) -> lds_a[lane]
//   2. global_load_async_to_lds_b32(src_b + lane)        -> lds_b[lane]
//   3. s_wait_asynccnt<0>()   — one wait must drain both
//   4. Read lds_a[lane] -> dst_a[lane], lds_b[lane] -> dst_b[lane]
//
// If s_wait_asynccnt only drained one instruction type, the wave would read
// LDS before the other write completed, producing stale data and a test FAIL.

struct ASYNCcntOrderingKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const int* __restrict__ src_a,
                                   const int* __restrict__ src_b,
                                   int* __restrict__ dst_a,
                                   int* __restrict__ dst_b,
                                   int mask) const
    {
#ifdef __gfx1250__
        __shared__ int lds_a[NUM_LANES];
        __shared__ int lds_b[NUM_LANES];

        const int lane_id = threadIdx.x;

        // Step 1: CLUSTER_LOAD_ASYNC_TO_LDS_B32 -> lds_a. Increments ASYNCcnt.
        ck_tile::cluster_multicast_load_async_to_lds(
            src_a + lane_id, ck_tile::to_lds(lds_a + lane_id), mask);

        // Step 2: GLOBAL_LOAD_ASYNC_TO_LDS_B32 -> lds_b. Also increments ASYNCcnt.
        __builtin_amdgcn_global_load_async_to_lds_b32(
            ck_tile::to_global(src_b + lane_id), ck_tile::to_lds(lds_b + lane_id), 0, 0);

        // Step 3: Single wait — must drain both async loads.
        ck_tile::s_wait_asynccnt<0>();

        // Step 4: Read both LDS slots. Correct data in both confirms shared
        // ASYNCcnt correctly tracks both instruction types.
        dst_a[lane_id] = lds_a[lane_id];
        dst_b[lane_id] = lds_b[lane_id];
#else
        (void)src_a;
        (void)src_b;
        (void)dst_a;
        (void)dst_b;
        (void)mask;
#endif
    }
};

void run_asynccnt_ordering_test(const std::vector<int>& h_src_a,
                                const std::vector<int>& h_src_b,
                                int mask,
                                const char* test_name)
{
    std::vector<int> h_dst_a(NUM_LANES), h_dst_b(NUM_LANES);

    ck_tile::DeviceMem d_src_a(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_src_b(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_dst_a(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_dst_b(NUM_LANES * sizeof(int));
    d_src_a.ToDevice(h_src_a.data());
    d_src_b.ToDevice(h_src_b.data());
    d_dst_a.SetBytePattern(0xFF);
    d_dst_b.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(ASYNCcntOrderingKernel{},
                                       dim3(1),
                                       dim3(1),
                                       dim3(NUM_LANES),
                                       0,
                                       static_cast<const int*>(d_src_a.GetDeviceBuffer()),
                                       static_cast<const int*>(d_src_b.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst_a.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst_b.GetDeviceBuffer()),
                                       mask);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst_a.FromDevice(h_dst_a.data());
    d_dst_b.FromDevice(h_dst_b.data());

    for(int i = 0; i < NUM_LANES; i++)
    {
        EXPECT_EQ(h_dst_a[i], h_src_a[i])
            << test_name << ": lds_a mismatch at lane " << i << " (cluster load: got " << h_dst_a[i]
            << ", want " << h_src_a[i] << ")";
        EXPECT_EQ(h_dst_b[i], h_src_b[i])
            << test_name << ": lds_b mismatch at lane " << i << " (global load: got " << h_dst_b[i]
            << ", want " << h_src_b[i] << ")";
    }
}

TEST(ASYNCcntOrdering, MixedAsyncLoads_B32_SingleWGP)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> src_a(NUM_LANES), src_b(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
    {
        src_a[i] = 1000 + i; // cluster load source -> lds_a
        src_b[i] = 2000 + i; // global load source  -> lds_b
    }
    run_asynccnt_ordering_test(src_a, src_b, 0x1, "MixedAsyncLoads_B32_SingleWGP");
}

// ---------------------------------------------------------------------------
// Group 6: PartialBroadcast — non-contiguous mask, mixed instruction types
// ---------------------------------------------------------------------------
// 4 WGPs, mask = 0x5 (binary 0101). WGPs 0 and 2 participate in the cluster
// multicast; WGPs 1 and 3 do not.
//
// Participating WGPs (flat_id bit set in mask):
//   cluster_multicast_load_async_to_lds(shared_src) -> lds_buf[lane]
//   Expected LDS: broadcast_val in every slot
//
// Non-participating WGPs (flat_id bit clear in mask):
//   global_load_async_to_lds_b32(g_src + lane) -> lds_buf[lane]
//   Expected LDS: 5000 + lane in every slot
//
// This simultaneously verifies:
//   1. Multicast data is delivered only to WGPs whose bits are set in M0 —
//      non-participating WGPs do not receive the broadcast value.
//   2. Both async instruction types coexist in the same cluster on the same
//      ASYNCcnt without cross-contaminating each other's LDS destinations.

struct PartialBroadcastKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const int* __restrict__ shared_src,
                                   const int* __restrict__ g_src,
                                   int* __restrict__ dst,
                                   int mask) const
    {
#ifdef __gfx1250__
        __shared__ int lds_buf[NUM_LANES];

        const int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();
        const int lane_id = threadIdx.x;

        if((mask >> flat_id) & 1)
        {
            // Participating WGP: cluster multicast load -> LDS.
            ck_tile::cluster_multicast_load_async_to_lds(
                shared_src, ck_tile::to_lds(lds_buf + lane_id), mask);
        }
        else
        {
            // Non-participating WGP: async global load -> LDS with sentinel values.
            __builtin_amdgcn_global_load_async_to_lds_b32(
                ck_tile::to_global(g_src + lane_id), ck_tile::to_lds(lds_buf + lane_id), 0, 0);
        }

        ck_tile::block_sync_lds_direct_load<0>();

        dst[blockIdx.x * blockDim.x + lane_id] = lds_buf[lane_id];
#else
        (void)shared_src;
        (void)g_src;
        (void)dst;
        (void)mask;
#endif
    }
};

void run_partial_broadcast_test(int num_wgs,
                                int mask,
                                int broadcast_val,
                                const std::vector<int>& h_g_src,
                                const char* test_name)
{
    const int total_threads = num_wgs * NUM_LANES;
    std::vector<int> h_dst(total_threads);

    ck_tile::DeviceMem d_shared_src(sizeof(int));
    ck_tile::DeviceMem d_g_src(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(int));
    d_shared_src.ToDevice(&broadcast_val);
    d_g_src.ToDevice(h_g_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(PartialBroadcastKernel{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(NUM_LANES),
                                       static_cast<std::size_t>(0),
                                       static_cast<const int*>(d_shared_src.GetDeviceBuffer()),
                                       static_cast<const int*>(d_g_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       mask);
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    for(int wgp = 0; wgp < num_wgs; wgp++)
    {
        const bool participating = (mask >> wgp) & 1;
        for(int lane = 0; lane < NUM_LANES; lane++)
        {
            const int got  = h_dst[wgp * NUM_LANES + lane];
            const int want = participating ? broadcast_val : h_g_src[lane];
            EXPECT_EQ(got, want) << test_name << ": WGP " << wgp << " lane " << lane
                                 << (participating ? " (cluster)" : " (global)") << ": got " << got
                                 << ", want " << want;
        }
    }
}

TEST(PartialBroadcast, B32_4WGP_Mask0x5)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    // mask = 0x5 = 0101: WGPs 0 and 2 receive broadcast, WGPs 1 and 3 do global load.
    std::vector<int> g_src(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        g_src[i] = 5000 + i;
    run_partial_broadcast_test(4, 0x5, static_cast<int>(0xDECAFBAD), g_src, "B32_4WGP_Mask0x5");
}

// ---------------------------------------------------------------------------
// Group 8: MultiWGP + LDSVisibility Combined — the canonical GEMM tile-load
// ---------------------------------------------------------------------------
// 4 WGPs in a cluster, 4 waves per WG (128 threads). Wave 0 of each WG issues
// cluster_multicast_load_async_to_lds (true broadcast: all lanes load from the
// same source address). After block_sync_lds_direct_load, waves 1–3 read from
// the same LDS buffer and write to global for host verification.
//
// This is the canonical GEMM prefetch pattern:
//   - One wave per WG issues the async cluster load (simulating a "load wave")
//   - All other waves in the WG read from LDS to compute (simulating "compute waves")
//
// Groups 2 and 4 test LDS visibility and multi-WGP broadcast in isolation;
// this group tests the combination. A bug where the barrier doesn't fence
// the async LDS write from wave 0 before waves 1–3 read would appear here
// but not in Groups 2 or 4 individually.
//
// Verification:
//   - Wave 0: each lane loaded src_val -> lds_buf[lane] (confirmed via dst)
//   - Waves 1–3: each lane read lds_buf[lane_id] = src_val (cross-wave visibility)
//   - All WGPs: same src_val in every LDS slot (multi-WGP broadcast)

template <typename T>
struct MultiWGPLDSVisibilityKernel
{
    static constexpr int kBlockSize = 4 * NUM_LANES; // 128 threads = 4 waves

    CK_TILE_DEVICE void operator()(const T* __restrict__ shared_src,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids,
                                   int num_wgs) const
    {
#ifdef __gfx1250__
        __shared__ T lds_buf[NUM_LANES]; // 32 slots; all waves in WG share this

        const int flat_id   = __builtin_amdgcn_cluster_workgroup_flat_id();
        const int thread_id = threadIdx.x;
        const int lane_id   = thread_id % NUM_LANES;
        const int wave_id   = thread_id / NUM_LANES;

        if(thread_id == 0)
            diag_ids[blockIdx.x] = flat_id;

        const int mask = (1 << num_wgs) - 1;

        if(wave_id == 0)
        {
            // Wave 0: broadcast src_val into every LDS slot via per-lane VDST.
            ck_tile::cluster_multicast_load_async_to_lds(
                shared_src, ck_tile::to_lds(lds_buf + lane_id), mask);
        }

        // All waves call block_sync_lds_direct_load: it issues s_wait_asynccnt
        // (a no-op for waves 1–3 whose count is already 0), then
        // s_barrier_signal/wait. Barrier ensures LDS is visible to all waves
        // before any wave reads from lds_buf.
        ck_tile::block_sync_lds_direct_load<0>();

        dst[blockIdx.x * blockDim.x + thread_id] = lds_buf[lane_id];
#else
        (void)shared_src;
        (void)dst;
        (void)diag_ids;
        (void)num_wgs;
#endif
    }
};

template <typename T>
void run_multiwgp_lds_visibility_test(int num_wgs, const T& src_val, const char* test_name)
{
    const int threads_per_wg = 4 * NUM_LANES;
    const int total_threads  = num_wgs * threads_per_wg;

    std::vector<T> h_dst(total_threads);
    std::vector<int> h_diag_ids(num_wgs);

    ck_tile::DeviceMem d_src(sizeof(T));
    ck_tile::DeviceMem d_dst(total_threads * sizeof(T));
    ck_tile::DeviceMem d_diag_ids(num_wgs * sizeof(int));
    d_src.ToDevice(&src_val);
    d_dst.SetBytePattern(0xFF);
    d_diag_ids.SetZero();

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(MultiWGPLDSVisibilityKernel<T>{},
                                       dim3(num_wgs, 1, 1),
                                       dim3(num_wgs),
                                       dim3(threads_per_wg),
                                       static_cast<std::size_t>(0),
                                       static_cast<const T*>(d_src.GetDeviceBuffer()),
                                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                                       static_cast<int*>(d_diag_ids.GetDeviceBuffer()),
                                       num_wgs);
    ASSERT_EQ(kernel(sc), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());
    d_diag_ids.FromDevice(h_diag_ids.data());

    // Verify flat IDs are contiguous (cluster layout assumption).
    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << test_name << ": blockIdx.x=" << i
                                    << " expected flat_id=" << i << " got " << h_diag_ids[i];

    // Every thread in every WGP must read src_val from LDS (waves 0–3, all WGPs).
    for(int wgp = 0; wgp < num_wgs; wgp++)
    {
        for(int wave = 0; wave < 4; wave++)
        {
            for(int lane = 0; lane < NUM_LANES; lane++)
            {
                const int idx = wgp * threads_per_wg + wave * NUM_LANES + lane;
                EXPECT_EQ(std::memcmp(&h_dst[idx], &src_val, sizeof(T)), 0)
                    << test_name << ": WGP " << wgp << " wave " << wave << " lane " << lane
                    << " mismatch";
            }
        }
    }
}

TEST(MultiWGPLDSVisibility, B32_4WGP_4Waves)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    run_multiwgp_lds_visibility_test<int>(4, static_cast<int>(0xDECAFBAD), "B32_4WGP_4Waves");
}

TEST(MultiWGPLDSVisibility, B128_4WGP_4Waves)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    int4 src_val = {static_cast<int>(0xDECAFBAD),
                    static_cast<int>(0xDEADBEEF),
                    0x12345678,
                    static_cast<int>(0xAAAAAAAA)};
    run_multiwgp_lds_visibility_test<int4>(4, src_val, "B128_4WGP_4Waves");
}

// ---------------------------------------------------------------------------
// Group 10: ConcurrentGroups — LDS routing isolation between independent groups
// ---------------------------------------------------------------------------
// 4 WGPs in one cluster, two independent broadcast groups:
//   WGPs 0/1: mask = 0x3, load val_a into LDS
//   WGPs 2/3: mask = 0xC, load val_b into LDS
//
// Each WGP branches on flat_id to determine its group, then issues
// cluster_multicast_load_async_to_lds with the appropriate mask and source.
//
// For CLUSTER_LOAD_B (VGPR destination), misdirected data would land in a
// per-thread VGPR that is private to one wave and physically unreadable by
// another WG — so VGPR tests cannot detect LDS routing bugs. Here, if the
// hardware routes val_a to WGPs 2/3's LDS (or vice versa), the host
// verification catches it. This is the only test that can expose such a bug.

template <typename T>
struct ConcurrentGroupsLDSKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const T* __restrict__ src_a,
                                   const T* __restrict__ src_b,
                                   T* __restrict__ dst,
                                   int* __restrict__ diag_ids) const
    {
#ifdef __gfx1250__
        __shared__ T lds_buf[NUM_LANES];

        const int flat_id = __builtin_amdgcn_cluster_workgroup_flat_id();
        const int lane_id = threadIdx.x;

        if(lane_id == 0)
            diag_ids[blockIdx.x] = flat_id;

        if(flat_id < 2)
        {
            // Group A: WGPs 0 and 1, mask = 0x3
            ck_tile::cluster_multicast_load_async_to_lds(
                src_a, ck_tile::to_lds(lds_buf + lane_id), 0x3);
        }
        else
        {
            // Group B: WGPs 2 and 3, mask = 0xC
            ck_tile::cluster_multicast_load_async_to_lds(
                src_b, ck_tile::to_lds(lds_buf + lane_id), 0xC);
        }

        ck_tile::s_wait_asynccnt<0>();

        dst[blockIdx.x * blockDim.x + lane_id] = lds_buf[lane_id];
#else
        (void)src_a;
        (void)src_b;
        (void)dst;
        (void)diag_ids;
#endif
    }
};

template <typename T>
void run_concurrent_groups_lds_test(const T& val_a, const T& val_b, const char* test_name)
{
    constexpr int num_wgs   = 4;
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
    auto kernel = ck_tile::make_kernel(ConcurrentGroupsLDSKernel<T>{},
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

    for(int i = 0; i < num_wgs; i++)
        EXPECT_EQ(h_diag_ids[i], i) << test_name << ": blockIdx.x=" << i
                                    << " expected flat_id=" << i << " got " << h_diag_ids[i];

    for(int wgp = 0; wgp < num_wgs; wgp++)
    {
        const T& expected = (wgp < 2) ? val_a : val_b;
        for(int lane = 0; lane < NUM_LANES; lane++)
        {
            const int idx = wgp * NUM_LANES + lane;
            EXPECT_EQ(std::memcmp(&h_dst[idx], &expected, sizeof(T)), 0)
                << test_name << ": WGP " << wgp << " lane " << lane << " mismatch"
                << " (expected group " << (wgp < 2 ? "A" : "B") << ")";
        }
    }
}

TEST(ConcurrentGroupsLDS, B32_4WGP_TwoGroups)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    run_concurrent_groups_lds_test<int>(
        static_cast<int>(0xAAAAAAAA), static_cast<int>(0xBBBBBBBB), "B32_4WGP_TwoGroups");
}

TEST(ConcurrentGroupsLDS, B128_4WGP_TwoGroups)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    int4 val_a = {
        static_cast<int>(0xAAAAAAAA), static_cast<int>(0x11111111), 0x22222222, 0x33333333};
    int4 val_b = {
        static_cast<int>(0xBBBBBBBB), static_cast<int>(0x44444444), 0x55555555, 0x66666666};
    run_concurrent_groups_lds_test<int4>(val_a, val_b, "B128_4WGP_TwoGroups");
}

// ---------------------------------------------------------------------------
// Group 11: BufferViewAsyncGet — cluster_async_get() through buffer_view
// ---------------------------------------------------------------------------
// Tests the buffer_view::cluster_async_get() interface, which wraps
// cluster_multicast_load_async_to_lds and handles global pointer arithmetic
// and address space casting internally.
//
// Test 1 (B32_BasicLoad): Verifies that cluster_async_get<int> loads
// src[i + linear_offset] into the per-lane LDS slot correctly, using the
// buffer_view's p_data_ as the global source base.
//
// Test 2 (B32_InstOffset): Verifies the ISA-specified behaviour of inst_offset.
// Per MI400 ISA (section 4.9.9.1), CLUSTER_LOAD_ASYNC_TO_LDS applies
// INST_OFFSET to BOTH the global source address (VADDR) and the LDS
// destination address (VDST):
//   LDS[VDST + INST_OFFSET] = GLOBAL[VADDR + INST_OFFSET]
// With inst_offset=4 (one int32) and per-lane VDST = &lds_buf[lane*2]:
//   - Source reads from src[lane+1]  (VADDR + 4 bytes)
//   - LDS writes to lds_buf[lane*2+1] (VDST + 4 bytes)
// The even slot (lds_buf[lane*2] = VDST) is left as sentinel.

using TestBufView = ck_tile::buffer_view<ck_tile::address_space_enum::global,
                                         int,
                                         ck_tile::index_t,
                                         true,
                                         ck_tile::amd_buffer_coherence_enum::coherence_default>;

// Kernel 1: basic load — each lane loads src[lane_id] into lds_buf[lane_id]
// via buffer_view::cluster_async_get.
struct BufferViewBasicKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const int* src, int* dst, int mask) const
    {
#ifdef __gfx1250__
        __shared__ int lds_buf[NUM_LANES];

        const int lane_id = threadIdx.x;

        TestBufView view;
        view.p_data_      = const_cast<int*>(src);
        view.buffer_size_ = NUM_LANES;

        view.template cluster_async_get<int>(lds_buf + lane_id, lane_id, 0, mask);

        ck_tile::s_wait_asynccnt<0>();

        dst[lane_id] = lds_buf[lane_id];
#else
        (void)src;
        (void)dst;
        (void)mask;
#endif
    }
};

TEST(BufferViewAsyncGet, B32_BasicLoad)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    std::vector<int> h_src(NUM_LANES), h_dst(NUM_LANES);
    for(int i = 0; i < NUM_LANES; i++)
        h_src[i] = 3000 + i;

    ck_tile::DeviceMem d_src(NUM_LANES * sizeof(int));
    ck_tile::DeviceMem d_dst(NUM_LANES * sizeof(int));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(BufferViewBasicKernel{},
                                       dim3(1),
                                       dim3(1),
                                       dim3(NUM_LANES),
                                       0,
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       0x1);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    for(int i = 0; i < NUM_LANES; i++)
        EXPECT_EQ(h_dst[i], h_src[i]) << "B32_BasicLoad: mismatch at lane " << i;
}

// Kernel 2: inst_offset=4 — each lane supplies VDST = &lds_buf[lane*2].
// ISA applies inst_offset to both VADDR and VDST:
//   LDS[VDST+4] = GLOBAL[VADDR+4]  →  lds_buf[lane*2+1] = src[lane+1]
// src is allocated with NUM_LANES+1 elements so lane 31 reads src[32] safely.
struct BufferViewInstOffsetKernel
{
    static constexpr int kBlockSize = NUM_LANES;

    CK_TILE_DEVICE void operator()(const int* src, int* dst, int mask) const
    {
#ifdef __gfx1250__
        __shared__ int lds_buf[2 * NUM_LANES];

        const int lane_id  = threadIdx.x;
        const int sentinel = 0xDEADBEEF;

        lds_buf[lane_id * 2]     = sentinel;
        lds_buf[lane_id * 2 + 1] = sentinel;
        ck_tile::s_wait_dscnt<0>();

        TestBufView view;
        view.p_data_      = const_cast<int*>(src);
        view.buffer_size_ = NUM_LANES + 1;

        // inst_offset=4: routes LDS write to lds_buf[lane*2+1] and
        // source read to src[lane+1].
        view.template cluster_async_get<int, 4>(lds_buf + lane_id * 2, lane_id, 0, mask);

        ck_tile::s_wait_asynccnt<0>();

        dst[lane_id * 2]     = lds_buf[lane_id * 2];
        dst[lane_id * 2 + 1] = lds_buf[lane_id * 2 + 1];
#else
        (void)src;
        (void)dst;
        (void)mask;
#endif
    }
};

TEST(BufferViewAsyncGet, B32_InstOffset)
{
    if(ck_tile::get_device_revision() == 0)
    {
        GTEST_SKIP() << "Cluster multicast load async to LDS is not supported on asicRevision=0";
    }
    // NUM_LANES+1 elements so lane 31 reads src[32] without OOB.
    std::vector<int> h_src(NUM_LANES + 1);
    std::vector<int> h_dst(2 * NUM_LANES);
    for(int i = 0; i <= NUM_LANES; i++)
        h_src[i] = 3000 + i;

    ck_tile::DeviceMem d_src((NUM_LANES + 1) * sizeof(int));
    ck_tile::DeviceMem d_dst(2 * NUM_LANES * sizeof(int));
    d_src.ToDevice(h_src.data());
    d_dst.SetBytePattern(0xFF);

    ck_tile::stream_config sc{};
    auto kernel = ck_tile::make_kernel(BufferViewInstOffsetKernel{},
                                       dim3(1),
                                       dim3(1),
                                       dim3(NUM_LANES),
                                       0,
                                       static_cast<const int*>(d_src.GetDeviceBuffer()),
                                       static_cast<int*>(d_dst.GetDeviceBuffer()),
                                       0x1);
    ck_tile::launch_and_check(sc, kernel);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    d_dst.FromDevice(h_dst.data());

    const int sentinel = 0xDEADBEEF;
    for(int i = 0; i < NUM_LANES; i++)
    {
        // Even slot (VDST): inst_offset skips this — sentinel must remain.
        EXPECT_EQ(h_dst[i * 2], sentinel)
            << "lane " << i << " even slot: expected sentinel, got " << h_dst[i * 2];

        // Odd slot (VDST+4): both source and LDS shifted by inst_offset.
        // Source reads src[lane+1]; write lands at lds_buf[lane*2+1].
        EXPECT_EQ(h_dst[i * 2 + 1], h_src[i + 1])
            << "lane " << i << " odd slot: expected src[" << (i + 1) << "]=" << h_src[i + 1]
            << ", got " << h_dst[i * 2 + 1];
    }
}
