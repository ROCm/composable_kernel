// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/env.hpp"

using ::ck::DeviceMem;
using F8DataType = ck::f8_t;

#if defined(__gfx125__)
__device__ constexpr int hint  = __ATOMIC_RELAXED;
__device__ constexpr int scope = __MEMORY_SCOPE_SYSTEM;
// BUG: duration = 0x8000 (sleep-forever) should not be used as the wave might never wake up if the
// s_monitor_sleep(duration) is called when MWAIT=0
__device__ constexpr short duration = static_cast<short>(1 << 15) - 1; // forever - 1 clock cycle
#endif

/// @param ptr points to a buffer of 4 F8 numbers
__global__ void gpu_ping(F8DataType* ptr, const int Num, int* runNum, bool ck_logging)
{
#if defined(__gfx125__)
    int run = 0;
    ptr[0]  = F8DataType{0x38};
    while(run++ < Num && !ck::fp8_is_nan(ptr[0]) && !ck::fp8_is_nan(ptr[1]))
    {
        while((__builtin_amdgcn_flat_load_monitor_b32(
                   static_cast<int*>(static_cast<void*>(ptr)), hint, scope) &
               0xFF) == 0x38)
        {
            __builtin_amdgcn_s_monitor_sleep(duration);
            if(ck_logging)
                printf("PING goes to sleep at run = %d.\n", run);
        }
        if(ptr[0] == F8DataType{0})
        {
            ptr[0] = F8DataType{0x38}; // send 1 back
            if(ck_logging)
                printf("PING 1\n");
            __builtin_amdgcn_s_sleep(10); // sleep to simulate workload
        }
        else
        {
            ptr[0] = F8DataType{0x7F}; // signal failure
            if(ck_logging)
                printf("PING receives incorrect value: %x.\n", ptr[0].data);
        }
    }
    *runNum = run;
#else
    if(ptr && runNum && ck_logging)
        *runNum = Num; // Dummy
#endif
}

/// @param ptr points to a buffer of 4 F8 numbers
__global__ void gpu_pong(F8DataType* ptr, const int Num, int* runNum, bool ck_logging)
{
#if defined(__gfx125__)
    int run = 0;
    while(run++ < Num && !ck::fp8_is_nan(ptr[0]) && !ck::fp8_is_nan(ptr[1]))
    {
        while((__builtin_amdgcn_flat_load_monitor_b32(
                   static_cast<int*>(static_cast<void*>(ptr)), hint, scope) &
               0xFF) == 0)
        {
            // Wait for the ping thread to set the value to 0x38
            __builtin_amdgcn_s_monitor_sleep(duration);
            if(ck_logging)
                printf("PONG goes to sleep at run = %d.\n", run);
        }

        if(ptr[0] == F8DataType{0x38})
        {
            ptr[0] = F8DataType{0}; // send 0 back
            if(ck_logging)
                printf("PONG 0\n");
            __builtin_amdgcn_s_sleep(20); // sleep to simulate workload
        }
        else
        {
            ptr[1] = F8DataType{0x7F}; // signal failure
            if(ck_logging)
                printf("PONG receives incorrect value: %x.\n", ptr[0].data);
        }
    }
    *runNum = run;
#else
    if(ptr && runNum && ck_logging)
        *runNum = Num; // Dummy
#endif
}

/**
 * @brief Test for monitor-mwait synchronization using a single cache line.
 *
 * This test launches two kernels: `gpu_ping` and `gpu_pong`, which
 * communicate through a shared buffer `A_d`. The `gpu_ping` kernel
 * updates the buffer and waits for a specific value, while the `gpu_pong`
 * kernel checks the buffer and updates it accordingly. The test verifies memory-based
 * synchronization by ensuring that both kernels complete their iterations and the values in the
 * buffer are as expected.
 *
 */

static void test_single_cacheline()
{
    const int Num = 10;

    DeviceMem A_d(4 * sizeof(F8DataType)); // Buffer will be updated and checked in 2 threads
    DeviceMem runNum(2 * sizeof(int));     // Used to keep iteration number for verification

    A_d.SetValue(0);
    runNum.SetValue(0);
    const bool ck_logging = ck::EnvIsEnabled(CK_ENV(CK_LOGGING));

    hipStream_t stream[2];
    HIP_CHECK_ERROR(hipStreamCreate(&stream[0]));
    HIP_CHECK_ERROR(hipStreamCreate(&stream[1]));

    hipLaunchKernelGGL(gpu_ping,
                       dim3(1),
                       dim3(1),
                       0,
                       stream[0],
                       static_cast<F8DataType*>(A_d.GetDeviceBuffer()),
                       Num,
                       static_cast<int*>(runNum.GetDeviceBuffer()),
                       ck_logging);
    HIP_CHECK_ERROR(hipGetLastError());

    hipLaunchKernelGGL(gpu_pong,
                       dim3(1),
                       dim3(1),
                       0,
                       stream[1],
                       static_cast<F8DataType*>(A_d.GetDeviceBuffer()),
                       Num,
                       static_cast<int*>(runNum.GetDeviceBuffer()) + 1,
                       ck_logging);
    HIP_CHECK_ERROR(hipGetLastError());

    HIP_CHECK_ERROR(hipStreamSynchronize(stream[0]));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream[1]));

    std::vector<int> runNumHost(2);
    runNum.FromDevice(runNumHost.data());

    ASSERT_EQ(runNumHost[0], Num + 1);
    ASSERT_EQ(runNumHost[1], Num + 1);

    std::vector<F8DataType> A_host(4);
    A_d.FromDevice(A_host.data());

    EXPECT_EQ(A_host[0], F8DataType{0x38});
    EXPECT_EQ(A_host[1], F8DataType{0});

    HIP_CHECK_ERROR(hipStreamDestroy(stream[0]));
    HIP_CHECK_ERROR(hipStreamDestroy(stream[1]));
}

TEST(SYNCHRONIZATION, MonitorMwaitSingleCacheline)
{
    hipDeviceProp_t props;
    HIP_CHECK_ERROR(hipSetDevice(0));
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, 0));

    if(props.major == 12 && props.minor == 5)
    {
        test_single_cacheline();
    }
    else
    {
        GTEST_SKIP() << "MonitorMwait test is only supported on gfx125X devices";
    }
}

__global__ void gpu_ping(int* ptrA,
                         int* ptrB,
                         const int expectedA0,
                         const int expectedA1,
                         const int toUpdateB0,
                         const int toUpdateB1,
                         const int Num,
                         int* runNum)
{
#if defined(__gfx125__)

    auto tid = threadIdx.x;
    if(tid >= 4)
        return; // Only 4 threads are used in this test

    int run   = 0;
    ptrB[tid] = toUpdateB0;

    while(run++ < Num)
    {
        while(__builtin_amdgcn_flat_load_monitor_b128(
                  reinterpret_cast<ck::int32x4_t*>(ptrA), hint, scope)[tid] != expectedA0)
        {
            __builtin_amdgcn_s_monitor_sleep(duration);
        }
        ptrB[tid] = toUpdateB1;

        while(__builtin_amdgcn_flat_load_monitor_b128(
                  reinterpret_cast<ck::int32x4_t*>(ptrA), hint, scope)[tid] != expectedA1)
        {
            __builtin_amdgcn_s_monitor_sleep(duration);
        }
        ptrB[tid] = toUpdateB0;
    }
    *runNum = run;
#else
    if(ptrA && ptrB && expectedA0 != expectedA1 && toUpdateB0 != toUpdateB1 && runNum)
        *runNum = Num; // Dummy
#endif
}

__global__ void gpu_pong(int* ptrB,
                         int* ptrA,
                         const int expectedB0,
                         const int expectedB1,
                         const int toUpdateA0,
                         const int toUpdateA1,
                         const int Num,
                         int* runNum)
{
#if defined(__gfx125__)

    auto tid = threadIdx.x;
    if(tid >= 4)
        return; // Only 4 threads are used in this test

    int run = 0;
    while(run++ < Num)
    {
        while(__builtin_amdgcn_flat_load_monitor_b128(
                  reinterpret_cast<ck::int32x4_t*>(ptrB), hint, scope)[tid] != expectedB0)
        {
            __builtin_amdgcn_s_monitor_sleep(duration);
        }
        ptrA[tid] = toUpdateA0;

        while(__builtin_amdgcn_flat_load_monitor_b128(
                  reinterpret_cast<ck::int32x4_t*>(ptrB), hint, scope)[tid] != expectedB1)
        {
            __builtin_amdgcn_s_monitor_sleep(duration);
        }
        ptrA[tid] = toUpdateA1;
    }
    *runNum = run;
#else
    if(ptrB && ptrA && expectedB0 != expectedB1 && toUpdateA0 != toUpdateA1 && runNum)
        *runNum = Num; // Dummy
#endif
}

static void test_multiple_cachelines()
{
    const int Num = 100;
    DeviceMem A_d(4 * sizeof(int)); // A buffer will be updated in stream 2 and checked in stream 1
    DeviceMem B_d(4 * sizeof(int)); // B buffer will be updated in stream 1 and checked in stream 2

    DeviceMem runNum(2 * sizeof(int)); // Used to keep iteration number for verification

    // Device memory is ALIGNSIZE aligned during allocation, so this can guarantee that A_d and B_d
    // have different cache lines as cache line size (usually 64) is much smaller than ALIGNSIZE.
    constexpr auto ALIGNSIZE = 4096;
    auto A_d_ptr             = reinterpret_cast<uintptr_t>(A_d.GetDeviceBuffer());
    auto B_d_ptr             = reinterpret_cast<uintptr_t>(B_d.GetDeviceBuffer());
    EXPECT_EQ(A_d_ptr % ALIGNSIZE, 0);
    EXPECT_EQ(B_d_ptr % ALIGNSIZE, 0);

    const auto distance =
        (B_d_ptr > A_d_ptr ? (B_d_ptr - A_d_ptr) : (A_d_ptr - B_d_ptr)) * sizeof(int);

    EXPECT_GE(distance, ALIGNSIZE);

    A_d.SetValue(0);
    B_d.SetValue(0);
    runNum.SetValue(0);

    hipStream_t stream[2];
    HIP_CHECK_ERROR(hipStreamCreate(&stream[0]));
    HIP_CHECK_ERROR(hipStreamCreate(&stream[1]));
    const int val[2][2] = {{11, 12}, {21, 22}};

    hipLaunchKernelGGL(gpu_ping,
                       dim3(1),
                       dim3(4), // 4 threads in each stream
                       0,
                       stream[0],
                       static_cast<int*>(A_d.GetDeviceBuffer()),
                       static_cast<int*>(B_d.GetDeviceBuffer()),
                       val[0][0],
                       val[0][1],
                       val[1][0],
                       val[1][1],
                       Num,
                       static_cast<int*>(runNum.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    hipLaunchKernelGGL(gpu_pong,
                       dim3(1),
                       dim3(4),
                       0,
                       stream[1],
                       static_cast<int*>(B_d.GetDeviceBuffer()),
                       static_cast<int*>(A_d.GetDeviceBuffer()),
                       val[1][0],
                       val[1][1],
                       val[0][0],
                       val[0][1],
                       Num,
                       static_cast<int*>(runNum.GetDeviceBuffer()) + 1);
    HIP_CHECK_ERROR(hipGetLastError());

    HIP_CHECK_ERROR(hipStreamSynchronize(stream[0]));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream[1]));

    std::vector<int> runNumHost(2);
    runNum.FromDevice(runNumHost.data());

    ASSERT_EQ(runNumHost[0], Num + 1);
    ASSERT_EQ(runNumHost[1], Num + 1);

    HIP_CHECK_ERROR(hipStreamDestroy(stream[0]));
    HIP_CHECK_ERROR(hipStreamDestroy(stream[1]));
}

TEST(SYNCHRONIZATION, MonitorMwaitMultipleCachelines)
{
    hipDeviceProp_t props;
    HIP_CHECK_ERROR(hipSetDevice(0));
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, 0));

    if(props.major == 12 && props.minor == 5)
    {
        test_multiple_cachelines();
    }
    else
    {
        GTEST_SKIP() << "MonitorMwait test is only supported on gfx125X devices";
    }
}
