// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#ifdef __gfx12__
__device__ void llvm_amdgcn_s_wait_dscnt(short cnt) __asm("llvm.amdgcn.s.wait.dscnt");
#endif
#endif

__device__ void block_sync_lds()
{
#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#if defined(__gfx12__)
    // Please note that the call to wait_dscnt(0) is a workaroud for bugs in ROCm 7.2 to 10.1 where
    // the release fences (which should have equivalent semantics, but, unlike a wait_dscnt,
    // actually are visible to the LLVM memory model) didn't always work correctly if they were the
    // first memory operation in a loop. This wait call prevents the removal of unneeded LDS waits
    // and should be removed once possible (see the pre-gfx12 case below).
    llvm_amdgcn_s_wait_dscnt(0);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
// Please note that these are a workaround for bugs in ROCm 7.2 to 10.1 where release fences at the
// top of loops wouldn't properly flush writes across a loop boundary. They are strictly worse for
// performance that the fences alone, since the compiler cannot drop them if they are redundant /
// there are no LDS operations outstanding. They should be removed once these compilers are no
// longer supported.
#if defined(__gfx11__)
    __builtin_amdgcn_s_waitcnt(0xfc07);
#else
    __builtin_amdgcn_s_waitcnt(0xc07f);
#endif
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#endif
#else
    __syncthreads();
#endif
}

__device__ void block_sync_lds_direct_load()
{
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
#elif defined(__gfx12__)
    asm volatile("\
    s_wait_loadcnt 0x0 \n \
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);
#else
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif
}

__device__ void block_sync_lds_async_load()
{
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
    __syncthreads();
#else
    // fall back
    block_sync_lds();
#endif
}

__device__ void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace ck
