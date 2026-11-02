// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Device test of workgroup_barrier_arrive / workgroup_barrier_wait (split_barrier.hpp).
//
// Every thread writes a step-tagged value to an LDS buffer, and after the barrier reads the value
// another thread wrote, in another wave. Two buffers alternate, so one barrier per step guards
// both hazards: the read after the write (RAW) and the next write into a buffer that was read
// two steps ago (WAR). The output keeps what every thread saw at every step, so a reordered or
// stale read shows up as a wrong element rather than cancelling out of a sum.
//
// The LDS writes are plain stores the compiler can see, so it cannot fold the reads to undef.
//
// Runs on every target. With the split barrier (gfx12) the arrive and the wait are separate
// instructions; elsewhere the wait is a full barrier and the arrive does nothing. On gfx12 this
// checks the protocol works, not that it is race-free: a missing drain is a timing race that one
// run may not hit.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace {

constexpr ck_tile::index_t kNumThreads = 512; // 16 waves on wave32, 8 on wave64

// Never zero, so an element nobody wrote stays distinguishable from a delivered value.
CK_TILE_HOST_DEVICE constexpr int32_t step_value(int32_t step, int32_t tid)
{
    return (step + 1) * 100000 + tid + 1;
}

// The thread whose value thread tid reads: a bijection that crosses wave boundaries.
CK_TILE_HOST_DEVICE constexpr int32_t partner(int32_t tid) { return (tid * 7 + 37) % kNumThreads; }

// Waves with an even id arrive with a TENSORcnt wait, as TDM fill issuers would; the rest
// without. The branch is wave-uniform.
CK_TILE_DEVICE void arrive_mixed()
{
    if(ck_tile::get_warp_id() % 2 == 0)
    {
        ck_tile::workgroup_barrier_arrive<0>();
    }
    else
    {
        ck_tile::workgroup_barrier_arrive();
    }
}

} // namespace

// Kernels have external linkage: in an anonymous namespace the host pass, which never sees the
// device call, rejects their operator() as an unused member function (-Werror).
namespace split_barrier_test {

// Arrive and wait in the same step, with independent work between them (overlapping the
// barrier).
struct arrive_then_wait_kernel
{
    static constexpr ck_tile::index_t kBlockSize = kNumThreads;

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out, int32_t num_steps) const
    {
        __shared__ int32_t buf[2][kNumThreads];
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
        uint32_t overlap  = static_cast<uint32_t>(tid);

        for(int32_t step = 0; step < num_steps; ++step)
        {
            buf[step % 2][tid] = step_value(step, tid);
            arrive_mixed();
            overlap = overlap * 3u + static_cast<uint32_t>(step); // register-only, in the window
            ck_tile::workgroup_barrier_wait();
            out[step * kNumThreads + tid] = buf[step % 2][partner(tid)];
        }
        out[num_steps * kNumThreads + tid] = static_cast<int32_t>(overlap);
    }
};

// The double-buffered K-loop form: publish buffer 0 in a prologue, wait at the loop top, refill
// the other buffer right after the wait, read this buffer, arrive, and wait once after the loop.
struct loop_top_wait_kernel
{
    static constexpr ck_tile::index_t kBlockSize = kNumThreads;

    CK_TILE_DEVICE void operator()(int32_t* __restrict__ out, int32_t num_steps) const
    {
        __shared__ int32_t buf[2][kNumThreads];
        const int32_t tid = static_cast<int32_t>(threadIdx.x);

        buf[0][tid] = step_value(0, tid);
        arrive_mixed();
        for(int32_t step = 0; step < num_steps; ++step)
        {
            ck_tile::workgroup_barrier_wait();
            if(step + 1 < num_steps)
            {
                buf[(step + 1) % 2][tid] = step_value(step + 1, tid);
            }
            out[step * kNumThreads + tid] = buf[step % 2][partner(tid)];
            arrive_mixed();
        }
        ck_tile::workgroup_barrier_wait();
    }
};

} // namespace split_barrier_test

namespace {

using split_barrier_test::arrive_then_wait_kernel;
using split_barrier_test::loop_top_wait_kernel;

int32_t overlap_value(int32_t tid, int32_t num_steps)
{
    uint32_t v = static_cast<uint32_t>(tid);
    for(int32_t step = 0; step < num_steps; ++step)
    {
        v = v * 3u + static_cast<uint32_t>(step);
    }
    return static_cast<int32_t>(v);
}

template <typename Kernel>
void run_and_check(int32_t num_steps, bool has_overlap_row)
{
    const int32_t rows = num_steps + (has_overlap_row ? 1 : 0);
    ck_tile::DeviceMem out_buf(rows * kNumThreads * sizeof(int32_t));
    out_buf.SetZero();

    ck_tile::launch_and_check(ck_tile::stream_config{},
                              ck_tile::make_kernel(Kernel{},
                                                   dim3(1),
                                                   dim3(kNumThreads),
                                                   0,
                                                   static_cast<int32_t*>(out_buf.GetDeviceBuffer()),
                                                   num_steps));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "kernel did not complete";

    std::vector<int32_t> out(rows * kNumThreads);
    out_buf.FromDevice(out.data());

    for(int32_t step = 0; step < num_steps; ++step)
    {
        for(int32_t tid = 0; tid < kNumThreads; ++tid)
        {
            ASSERT_EQ(out[step * kNumThreads + tid], step_value(step, partner(tid)))
                << "steps " << num_steps << ": step " << step << " thread " << tid;
        }
    }
    if(has_overlap_row)
    {
        for(int32_t tid = 0; tid < kNumThreads; ++tid)
        {
            ASSERT_EQ(out[num_steps * kNumThreads + tid], overlap_value(tid, num_steps))
                << "steps " << num_steps << ": overlap value of thread " << tid;
        }
    }
}

class SplitBarrierSteps : public ::testing::TestWithParam<int32_t>
{
};

TEST_P(SplitBarrierSteps, ArriveThenWaitDeliversEveryStep)
{
    run_and_check<arrive_then_wait_kernel>(GetParam(), true);
}

TEST_P(SplitBarrierSteps, LoopTopWaitWithRefillDeliversEveryStep)
{
    run_and_check<loop_top_wait_kernel>(GetParam(), false);
}

// 1 and 2 steps never reuse a buffer; 3 and more do, and odd and even counts end on different
// buffers. The long runs catch a missing barrier in the rocJITsu emulator: a build with the
// barrier instructions removed fails the 64- and 256-step cases there, and passes the rest.
INSTANTIATE_TEST_SUITE_P(StepCounts, SplitBarrierSteps, ::testing::Values(1, 2, 3, 8, 9, 64, 256));

} // namespace
