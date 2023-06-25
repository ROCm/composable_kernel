// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {

// Initialization flag of Barrier object, can be any value except for zero
static constexpr int BarrierInitFlag = 0x7856;

#if defined(__gfx90a__)

// only the first thread-block in the synchronizaton group is supposed to call this function
__device__ inline void gms_init(int NumWarps, int* p_control_bits)
{
    int wave_id = get_wave_id();

    // clang-format off
    // regs[0] = BarrierInitFlag, regs[1] = NumWorkgroup, regs[2] = 0, regs[3] = 0
    // regs[0:3] using s[16:19]
    __asm__ volatile("s_cmp_lg_i32 %3, 0                           \n\
                      s_cbranch_scc1  skip_gms_init%=              \n\
                          s_movk_i32 s16, %1                          \n\
                          s_mov_b32 s17, %2                           \n\
                          s_movk_i32 s18, 0                           \n\
                          s_movk_i32 s19, 0                           \n\
                          s_atomic_cmpswap_x2 s[16:19], %0, 0         \n\
                          s_waitcnt lgkmcnt(0)                        \n\
                      skip_gms_init%=:"
                      :
                      : "s"(p_control_bits), "i"(BarrierInitFlag), "s"(NumWarps), "s"(wave_id)
                      : "s16", "s17", "s18", "s19");
    // clang-format on
};

// all the warps in the synchronization group is supposed to call this function
__device__ inline void gms_barrier(int* p_control_bits)
{
    // clang-format off
    __asm__ volatile("wait_initialized%=:                         \n\
                            s_load_dword s16, %0, 0 glc               \n\
                            s_waitcnt lgkmcnt(0)                      \n\
                            s_cmp_lg_u32 s16 %1                       \n\
                            s_cbranch_scc1 wait_initialized%=         \n\
                            s_atomic_sub %3, %0, %2                   \n\
                      wait_all_arrive%=:                          \n\
                            s_load_dword s17, %0, %2 glc              \n\
                            s_waitcnt lgkmcnt(0)                      \n\
                            s_cmp_lg_u32 s17, 0                       \n\
                            s_cbranch_scc1 wait_all_arrive%=          \n\
                      skip_barrier%=:"
                      :
                      : "s"(p_control_bits), "i"(BarrierInitFlag), "i"(sizeof(int)), "s"(1)
                      : "s16", "s17");
    // clang-format on
};

// only the first thread-block in the synchronizaton group is supposed to call this function
__device__ inline void gms_reset(int* p_control_bits)
{
    int wave_id = get_wave_id();

    // clang-format off
    // regs[0] = 0, regs[1] = BarrierInitFlag
    // regs[0:1] using s[16:17]
    __asm__ volatile("s_cmp_lg_i32 %2, 0                             \n\
                      s_cbranch_scc1 skip_gms_reset%=                \n\
                         s_movk_i32 s16, 0                              \n\
                         s_movk_i32 s17, %1                             \n\
                         s_atomic_cmpswap s[16:17], %0, 0               \n\
                         s_waitcnt lgkmcnt(0)                           \n\
                      skip_gms_reset%=:"
                      :
                      : "s"(p_control_bits), "i"(BarrierInitFlag), "s"(wave_id)
                      : "s16", "s17");
    // clang-format on
};

#else

// only the first thread-block in the synchronizaton group is supposed to call this function
__device__ inline void gms_init(int NumWarps, int* p_control_bits)
{
    union
    {
        int two32[2];
        unsigned long one64;
    } regs;

    regs.two32[0] = BarrierInitFlag;
    regs.two32[1] = NumWarps;

    if(threadIdx.x == 0)
        atomicCAS(reinterpret_cast<unsigned long*>(p_control_bits), 0, regs.one64);
};

// all the workgroups in the synchronization group is supposed to call this function
__device__ inline void gms_barrier(int* p_control_bits)
{
    constexpr int mask = warpSize - 1;

    if((threadIdx.x & mask) == 0)
    {
        // ensure the barrier object is initialized
        do
        {
            const int r0 = __atomic_load_n(&p_control_bits[0], __ATOMIC_RELAXED);

            if(r0 == BarrierInitFlag)
                break;

        } while(true);

        // go ahead toward the barrier line
        atomicSub(&p_control_bits[1], 1);

        // wait until all warps have arrived
        do
        {
            const int r1 = __atomic_load_n(&p_control_bits[1], __ATOMIC_RELAXED);

            if(r1 == 0)
                break;

        } while(true);
    };
};

// only the first thread-block in the synchronizaton group is supposed to call this function
__device__ inline void gms_reset(int* p_control_bits)
{
    // reset the barrier object
    if(threadIdx.x == 0)
        (void)atomicCAS(&p_control_bits[0], BarrierInitFlag, 0);
};

#endif

} // namespace ck
