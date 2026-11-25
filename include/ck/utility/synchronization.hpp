// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/number.hpp"

namespace ck {

#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#if defined(__gfx12__)
__device__ void llvm_amdgcn_s_wait_dscnt(short cnt) __asm("llvm.amdgcn.s.wait.dscnt");
#endif
#endif

#if CK_HAS_ASYNCMARK
namespace detail {
// Newer compilers take a trailing stage mask operand in the asyncmark builtins. MaskType holds its
// type for those and is empty otherwise, which drops the mask: such compilers keep one sequence
// of marks for all stages, so after each wait at least the async operations that the stage mask
// asks for have completed.
template <typename... MaskType>
struct asyncmark_builtins
{
    template <unsigned int StageMask>
    __device__ static void mark()
    {
        __builtin_amdgcn_asyncmark(MaskType{StageMask}...);
    }

    template <index_t NumMarks, unsigned int StageMask>
    __device__ static void wait()
    {
        __builtin_amdgcn_wait_asyncmark(NumMarks, MaskType{StageMask}...);
    }
};

// Overload resolution selects the arity the builtins accept. The probe sits in a function
// template's return type because clang checks AMDGPU builtin calls against an enclosing function,
// which partial specializations and default template arguments lack.
template <typename StageMask>
auto select_asyncmark_builtins(int) -> decltype(__builtin_amdgcn_asyncmark(StageMask{}),
                                                asyncmark_builtins<StageMask>{});

template <typename StageMask>
asyncmark_builtins<> select_asyncmark_builtins(long);

using asyncmark_builtins_t = decltype(select_asyncmark_builtins<unsigned int>(0));
} // namespace detail

// Appends an asyncmark to the sequence of each stage in StageMask, a mask of LLVM's async stage
// bits in which zero names every stage.
template <unsigned int StageMask = 0>
__device__ void asyncmark()
{
    detail::asyncmark_builtins_t::template mark<StageMask>();
}

// Waits until the sequence of each stage in StageMask, where zero names every stage, holds at
// most NumMarks asyncmarks.
template <index_t NumMarks = 0, unsigned int StageMask = 0>
__device__ void wait_asyncmark()
{
    detail::asyncmark_builtins_t::template wait<NumMarks, StageMask>();
}
#endif

template <index_t cnt = 0>
__device__ void wait_dscnt(Number<cnt> = {})
{
#if defined(__gfx12__)
    llvm_amdgcn_s_wait_dscnt(cnt);
#elif defined(__gfx11__) || defined(__gfx117__)
    constexpr index_t full_vmcnt  = 0x3F << 10;        // [15:10]
    constexpr index_t full_expcnt = 0x07;              // [2:0]
    constexpr index_t ds_cnt      = (cnt & 0x3F) << 4; // [9:4]
    constexpr index_t waitcnt     = full_vmcnt | full_expcnt | ds_cnt;
    __builtin_amdgcn_s_waitcnt(waitcnt);
#else
    constexpr index_t full_vmcnt = ((0x3F & 0xF) << 0) | ((0x3F & 0x30) << 10); // split: low4 + hi2
    constexpr index_t full_expcnt = 0x07 << 4;                                  // [6:4]
    constexpr index_t ds_cnt      = (cnt & 0x0F) << 8;                          // [11:8]
    constexpr index_t waitcnt     = full_vmcnt | full_expcnt | ds_cnt;
    __builtin_amdgcn_s_waitcnt(waitcnt);
#endif
}

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
#if defined(__gfx11__) || defined(__gfx117__)
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
#if CK_HAS_ASYNCMARK
    // Wait on async traffic to conclude, then sync the workgroup so that everyone sees the values
    // in LDS.
    wait_asyncmark();
    // Inline assembly is used so wait asyncnct needs to be called explicitly on gfx1250
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
#endif
    block_sync_lds();
#else
#if defined(__gfx125__)
    // The s_wait_asynccnt and manual barrier builtins are invisible to the LLVM memory model, so
    // wrap them in release/acquire fences to ensure nothing is reordered across these waits.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_wait_asynccnt(0);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
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
#endif
}

// Completes the outstanding asynchronous global->LDS DMA loads and publishes the LDS to the
// workgroup with a barrier. On the asyncmark path this is arch-agnostic -- wait_asyncmark(0) drains
// the async loads and is visible to the LLVM memory model (no manual fence/waitcnt needed other
// than what you get from block_sync_lds), then block_sync_lds() barriers. The #else fallbacks
// reproduce the pre-asyncmark behavior per arch.
__device__ void block_sync_lds_async_load()
{
#if CK_HAS_ASYNCMARK
    wait_asyncmark();
    block_sync_lds();
#elif defined(__gfx125__)
    // Fallback: async-engine completion wait + "local" fences + manual barrier (avoids the
    // redundant s_wait_loadcnt_dscnt that __syncthreads would inject).
    __builtin_amdgcn_s_wait_asynccnt(0);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    // Fallback (gfx9, no asyncmark): explicit vmcnt(0) + "global" ordering fence + LDS barrier.
    __builtin_amdgcn_s_waitcnt(3952); // vmcnt(0)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
    block_sync_lds();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
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
