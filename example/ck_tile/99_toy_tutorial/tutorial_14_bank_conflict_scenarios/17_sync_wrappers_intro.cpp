// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.17: Synchronization wrappers and async-engine helpers
 *
 * Tutorial 14.16 covered the five "raw" AMDGCN primitives that
 * unified_attention_pipeline.hpp:1054-1055 boils down to:
 *   __builtin_amdgcn_s_barrier
 *   __builtin_amdgcn_s_waitcnt   (via s_waitcnt_vmcnt / s_waitcnt_lgkmcnt)
 *   __builtin_amdgcn_s_setprio
 *   __builtin_amdgcn_sched_barrier
 *   ASM_MARKER(...)
 *
 * Most production code in ck_tile does NOT call those builtins directly.
 * It calls one of the convenience wrappers in arch.hpp / utility.hpp /
 * amd_buffer_addressing*.hpp that compose them into one of a handful of
 * standard idioms. This tutorial walks each wrapper in isolation, with
 * a one-line note on what raw primitive it folds into.
 *
 *   PART A - block_sync_lds<lgkm=0>()
 *            == s_waitcnt lgkmcnt(N) + __builtin_amdgcn_s_barrier
 *            The everyday "I just wrote LDS, sync the workgroup before
 *            anyone reads it" call. Re-runs Tutorial 14.16 PART A
 *            "correct" variant, just expressed via the wrapper.
 *
 *   PART B - block_sync_lds_direct_load<vm=0>()
 *            == s_waitcnt vmcnt(N) + __builtin_amdgcn_s_barrier
 *            This wrapper IS the line-1054 idiom. Used after async
 *            "buffer_load ... lds" stages where the LDS write is
 *            bundled into the global op and only vmcnt moved.
 *
 *   PART C - workgroup_barrier::wait_eq_wave (cross-WORKGROUP sync)
 *            Spin-loop on a global counter using __builtin_amdgcn_s_sleep
 *            for power-friendly polling. NOT a wave/block barrier --
 *            this is the tool for ordering DIFFERENT workgroups via
 *            global memory (e.g. split-K reductions). Distinct
 *            mechanism from s_barrier.
 *
 *   Closing comments cover the wrappers that are mostly invisible from
 *   user code:
 *     async_buffer_load_fence      vmcnt-only wait, no barrier
 *     s_nop                        N empty cycles, used to widen issue gaps
 *     __builtin_amdgcn_s_sleep     wave power-down (used inside wait_eq_wave)
 *     m0_set_with_memory           programs m0 before async buffer_load lds
 *     __builtin_amdgcn_iglp_opt    competing alternative to sched_barrier
 *
 * Build:
 *   target is aa_tutorial_14_17_sync_wrappers_intro
 */

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/workgroup_barrier.hpp"

#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

using namespace ck_tile;

// Same marker macro used by tutorial 14.16. Lets us locate the sync
// sequences in the AMDGCN dump by grep for "[POYENC]". The two
// sched_barrier(0)'s bracket the asm("...") line so the LLVM scheduler
// is forbidden to move surrounding instructions across the marker.
#define ASM_MARKER_DEMO(name)            \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #name);   \
    __builtin_amdgcn_sched_barrier(0);

// --------------------------------------------------------------------------
// PART A: block_sync_lds<0>() instead of the manual pair
//
// Reference: include/ck_tile/core/arch/arch.hpp:1060-1064
//
//     template <index_t lgkmcnt = 0>
//     CK_TILE_DEVICE void block_sync_lds()
//     {
//         s_waitcnt_barrier<kMaxVmCnt, kMaxExpCnt, lgkmcnt>();
//     }
//
//   == s_waitcnt lgkmcnt(N) followed by __builtin_amdgcn_s_barrier()
//
// The "<0>" template argument is the lgkmcnt threshold to wait for
// (default 0 = drain all outstanding LDS ops).
//
// Usage rule of thumb: any time you do
//      lds[tid] = something;
//      ... read lds across threads ...
// put a block_sync_lds<0>() between them.
// --------------------------------------------------------------------------

__global__ void partA_block_sync_lds_kernel(int* output)
{
    __shared__ int lds[1];
    const int tid = threadIdx.x;

    if(tid == 0)
        lds[0] = 42;

    // Single line replaces both s_waitcnt_lgkmcnt<0>() and __builtin_amdgcn_s_barrier().
    block_sync_lds<0>();

    output[tid] = lds[0];
}

// --------------------------------------------------------------------------
// PART B: block_sync_lds_direct_load<0>() == the line-1054 idiom
//
// Reference: include/ck_tile/core/arch/arch.hpp:1066-1070
//
//     template <index_t vmcnt = 0>
//     CK_TILE_DEVICE void block_sync_lds_direct_load()
//     {
//         s_waitcnt_barrier<vmcnt, kMaxExpCnt, kMaxLgkmCnt>();
//     }
//
//   == s_waitcnt vmcnt(N) followed by __builtin_amdgcn_s_barrier()
//
// This is BIT-FOR-BIT what unified_attention_pipeline.hpp:1054-1055 does:
//
//     s_waitcnt_vmcnt<0>();
//     __builtin_amdgcn_s_barrier();
//
// The pipeline spells the pair out manually only because the surrounding
// code is hand-tuned to keep an unrelated lgkmcnt outstanding across
// the barrier (which the wrapper would mask out). When you don't have
// that constraint, prefer the wrapper -- it's shorter, less error-prone,
// and per-arch correct (gfx12 swaps in s_barrier_signal/s_barrier_wait
// automatically; the manual pair would have to be #ifdef'd).
//
// Demo: same global -> VGPR -> LDS -> cross-thread read pattern as
//       Tutorial 14.16 PART B, but expressed through the wrappers.
// --------------------------------------------------------------------------

constexpr int kPartBSize = 64;

__global__ void partB_wrapped_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kPartBSize];
    const int tid = threadIdx.x;

    int v = g_in[tid]; // global load -> vmcnt

    // We are about to USE v in a VGPR (to write it to LDS). If we had
    // also issued the load via the async "buffer_load ... lds" path,
    // the LDS write would be bundled into the global op and there
    // would be NO ds_write to wait on -- only vmcnt. That is the
    // scenario block_sync_lds_direct_load is designed for.
    //
    // Here we keep the demo portable by going through a VGPR, which
    // means the compiler will inject its own vmcnt wait before this
    // line executes. Pretend that wait isn't there for narrative
    // purposes; the PATTERN is what matters.

    lds[tid] = v;

    // The line-1054 idiom, wrapped:
    ASM_MARKER_DEMO(partB_wrapped_sync_begin)
    block_sync_lds<0>(); // here we wait on lgkmcnt because we did a real ds_write.
                         // If v had landed via async buffer_load lds, this would
                         // instead be: block_sync_lds_direct_load<0>();
    ASM_MARKER_DEMO(partB_wrapped_sync_end)

    int neighbor = lds[(tid + 1) % kPartBSize];

    g_out[tid] = neighbor;
}

// Companion kernel that uses block_sync_lds_direct_load<0>() in the
// "right" spot so you can grep for it in the AMDGCN dump and see it
// compiles to exactly the same s_waitcnt vmcnt(0) + s_barrier pair as
// the manual code on line 1054 of the pipeline.
__global__ void partB_direct_load_shape_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kPartBSize];
    const int tid = threadIdx.x;

    int v    = g_in[tid];
    lds[tid] = v;

    // Even though we just did a normal ds_write (lgkmcnt-bumping), we
    // call the vmcnt-only wrapper here purely so the disassembler shows
    // 's_waitcnt vmcnt(0)' followed by 's_barrier' in the dump. In real
    // code this would only be correct if v had landed via
    // async_buffer_load + buffer_load_lds (no ds_write on our side).
    // See partB_async_buffer_load_lds_kernel below for the legitimate
    // use, and partB_direct_load_shape_multi_wave_kernel below for a
    // race repro that tries to expose the bug at runtime.
    ASM_MARKER_DEMO(partB_direct_load_sync_begin)
    block_sync_lds_direct_load<0>();
    ASM_MARKER_DEMO(partB_direct_load_sync_end)

    int neighbor = lds[(tid + 1) % kPartBSize];
    g_out[tid]   = neighbor;
}

// --------------------------------------------------------------------------
// partB_direct_load_shape_multi_wave_kernel
//
// Same buggy pattern as partB_direct_load_shape_kernel above, but with
// 128 threads = TWO waves on gfx9, and a cross-WAVE consumer pattern:
// lane tid in wave 0 reads lds[(tid + 64) % 128], a slot that lane
// (tid + 64) of wave 1 wrote. So the read genuinely depends on the
// OTHER wave's outstanding ds_write being globally visible by the time
// the s_barrier releases.
//
// With the wrong wrapper -- block_sync_lds_direct_load<0>() drains
// vmcnt only, not lgkmcnt -- this is undefined behaviour per the ISA:
// the s_barrier may release before lane (tid+64)'s ds_write has
// committed to LDS. In PRACTICE on gfx9/gfx950 the s_barrier hardware
// rendezvous happens to wait long enough for in-flight ds_writes, so
// this kernel will typically still produce correct output. The point
// of the kernel is to make that "happens to work" explicit:
//   - the runtime check passes
//   - the .s shows there is no s_waitcnt lgkmcnt(0) before the barrier
// On a stricter arch (gfx12 split barriers, gfx10/11 in some scheduler
// states) the same code would corrupt the LDS read.
// --------------------------------------------------------------------------

constexpr int kPartBMultiWaveSize = 128;

__global__ void partB_direct_load_shape_multi_wave_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kPartBMultiWaveSize];
    const int tid = threadIdx.x; // 0..127

    int v    = g_in[tid];
    lds[tid] = v; // ds_write -- bumps lgkmcnt, NOT drained by the wrapper below

    ASM_MARKER_DEMO(partB_multi_wave_sync_begin)
    block_sync_lds_direct_load<0>(); // drains vmcnt only (wrong for ds_write!)
    ASM_MARKER_DEMO(partB_multi_wave_sync_end)

    // Cross-wave read: lane tid reads what lane (tid+64) wrote.
    int neighbor = lds[(tid + 64) % kPartBMultiWaveSize];
    g_out[tid]   = neighbor;
}

// --------------------------------------------------------------------------
// partB_async_buffer_load_lds_kernel
//
// THE legitimate use of block_sync_lds_direct_load<0>(). This is the
// shape unified_attention_pipeline.hpp uses around line 1054.
//
// Setup: a "buffer_load_dword ... lds" instruction reads from global
// memory into LDS in ONE bundled op. The compute path on the producer
// side never executes a ds_write -- the LDS write is folded into the
// global buffer_load and only bumps vmcnt.
//
// The three pieces of plumbing:
//   1. make_wave_buffer_resource(ptr) builds the 128-bit V# (buffer
//      resource descriptor) that the buffer_load instruction uses to
//      address global memory.
//   2. m0_set_with_memory(byte_offset) writes the LDS-side BYTE
//      offset where this wave's slice of LDS begins. Each active lane
//      will end up writing to LDS[m0 + lane_id * sizeof(load)].
//   3. async_buffer_load_dwordxn_v<num_dwords> emits the actual
//      "buffer_load_dwordxN ... offen offset:N lds" inline-asm.
//      voffset is the per-lane byte offset into global; ioffset is the
//      compile-time immediate offset.
//
// After the issue, ONLY vmcnt has moved (no ds_write happened). The
// canonical drain is therefore vmcnt-only:
//   block_sync_lds_direct_load<0>()
//     == s_waitcnt vmcnt(0); s_barrier;
//
// That is exactly what the unified attention pipeline spells out by
// hand at lines 1054-1055.
// --------------------------------------------------------------------------

constexpr int kPartBAsyncSize = 64;

__global__ void partB_async_buffer_load_lds_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kPartBAsyncSize];
    const int tid = threadIdx.x;

    // Buffer resource descriptor (V#) for g_in.
    int32x4_t rsrc = make_wave_buffer_resource(g_in, kPartBAsyncSize * sizeof(int));

    // LDS-side base offset for this wave. Lane tid will end up writing
    // to LDS[m0 + tid * sizeof(int)] = lds[tid].
    m0_set_with_memory(0);

    // Issue the bundled global -> LDS load. No ds_write; only vmcnt
    // bumps. Marker brackets so the issue is locatable in the dump.
    ASM_MARKER_DEMO(partB_async_load_issue)
    async_buffer_load_dwordxn_v<1>(/*smem=*/lds,
                                   rsrc,
                                   /*voffset=*/tid * static_cast<index_t>(sizeof(int)),
                                   /*soffset=*/0,
                                   /*ioffset=*/0);

    // The line-1054 idiom in its proper habitat: drain vmcnt and
    // workgroup-barrier. lgkmcnt is already 0 because no ds op
    // happened yet, so the wrapper masking it out is exactly right.
    ASM_MARKER_DEMO(partB_async_sync_begin)
    block_sync_lds_direct_load<0>();
    ASM_MARKER_DEMO(partB_async_sync_end)

    // Cross-thread LDS read; the post-barrier ds_read is the only ds
    // op in this kernel.
    //
    // Aside: the compiler doesn't see that the buffer_load_dword ... lds
    // inline asm in async_buffer_load_dwordxn_v writes into our `lds`
    // array (it only ties the asm to a dummy pointer output via "=r").
    // Without forcing the read, the optimizer DCE's the whole post-sync
    // body and even drops the LDS allocation. Reading through a
    // `volatile` view defeats that without changing the demo's intent.
    volatile int* vlds = lds;
    int neighbor       = vlds[(tid + 1) % kPartBAsyncSize];
    g_out[tid]         = neighbor;
}

// --------------------------------------------------------------------------
// PART C: workgroup_barrier::wait_eq_wave (cross-workgroup sync)
//
// Reference: include/ck_tile/core/arch/workgroup_barrier.hpp:11-93
//
// workgroup_barrier wraps a uint32_t global counter and lets one block
// spin-wait on a value written by another block. The "wave" variant
// uses __builtin_amdgcn_s_sleep(1) between polls so the spinning wave
// gives up its issue slots to other waves on the same CU.
//
// This is a fundamentally different mechanism from __builtin_amdgcn_s_barrier:
//
//   s_barrier             | INSIDE one workgroup (block).
//                         | Synchronizes the waves of one block.
//                         | One SOPP, free.
//
//   workgroup_barrier     | BETWEEN workgroups.
//                         | Implemented in global memory via atomicAdd
//                         | + atomic load. Costs HBM round trips.
//                         | Used for split-K reductions, persistent
//                         | kernels, multi-stage CTAs that must agree
//                         | across the grid.
//
// Demo: two blocks. Block 0 writes a payload to global memory, then
//       calls b.inc() (== atomicAdd of 1 on the counter). Block 1 spins
//       on b.wait_eq_wave(1), then reads the payload safely.
//
// IMPORTANT: this only works because of HIP's memory model semantics:
// atomicAdd has acquire/release-ish ordering, and __atomic_load_n with
// __ATOMIC_RELAXED inside ld() pairs with that. For real cross-block
// publishing in production code you'd add a __threadfence_system() or
// equivalent. We keep it simple here; on standard MI200/MI300 hardware
// the demo will pass.
// --------------------------------------------------------------------------

constexpr int kPartCThreads = 64;

__global__ void partC_kernel(uint32_t* counter, int* payload, int* g_out)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    workgroup_barrier b(counter);

    if(bid == 0)
    {
        // Block 0 = producer.
        payload[tid] = tid * 7;
        // inc() does __syncthreads() then thread 0 atomicAdd(counter, 1).
        b.inc();
    }
    else
    {
        // Block 1 = consumer.
        // Polite spin until counter == 1, with __builtin_amdgcn_s_sleep
        // between polls. After this returns, payload[*] is safe to read.
        b.wait_eq_wave(1);

        g_out[tid] = payload[tid];
    }
}

// --------------------------------------------------------------------------
// Host driver
// --------------------------------------------------------------------------

static bool run_partA()
{
    constexpr int kThreads = 64;
    int* d_out             = nullptr;
    (void)hipMalloc(&d_out, kThreads * sizeof(int));
    (void)hipMemset(d_out, 0xCD, kThreads * sizeof(int));

    hipLaunchKernelGGL(
        partA_block_sync_lds_kernel, dim3(1), dim3(kThreads), 0, nullptr, d_out);
    auto err = hipDeviceSynchronize();

    std::vector<int> h_out(kThreads);
    (void)hipMemcpy(h_out.data(), d_out, kThreads * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_out);

    int wrong = 0;
    for(int v : h_out)
        if(v != 42)
            ++wrong;

    printf("[A block_sync_lds        ] tid 0 = %d, tid 63 = %d, wrong = %d/%d (err=%s)\n",
           h_out[0],
           h_out.back(),
           wrong,
           kThreads,
           hipGetErrorString(err));
    return wrong == 0 && err == hipSuccess;
}

static bool run_partB(bool use_direct_load_shape)
{
    int* d_in  = nullptr;
    int* d_out = nullptr;
    std::vector<int> h_in(kPartBSize);
    for(int i = 0; i < kPartBSize; ++i)
        h_in[i] = i * 10;

    (void)hipMalloc(&d_in, kPartBSize * sizeof(int));
    (void)hipMalloc(&d_out, kPartBSize * sizeof(int));
    (void)hipMemcpy(d_in, h_in.data(), kPartBSize * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0xCD, kPartBSize * sizeof(int));

    if(use_direct_load_shape)
        hipLaunchKernelGGL(partB_direct_load_shape_kernel,
                           dim3(1),
                           dim3(kPartBSize),
                           0,
                           nullptr,
                           d_in,
                           d_out);
    else
        hipLaunchKernelGGL(
            partB_wrapped_kernel, dim3(1), dim3(kPartBSize), 0, nullptr, d_in, d_out);

    auto err = hipDeviceSynchronize();

    std::vector<int> h_out(kPartBSize);
    (void)hipMemcpy(h_out.data(), d_out, kPartBSize * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_in);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < kPartBSize; ++i)
    {
        const int expected = ((i + 1) % kPartBSize) * 10;
        if(h_out[i] != expected)
            ++mismatches;
    }

    printf("[B %-26s] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           use_direct_load_shape ? "block_sync_lds_direct_load" : "block_sync_lds (wrapped)",
           h_out[0],
           h_out[1],
           h_out[2],
           h_out[3],
           mismatches,
           kPartBSize,
           hipGetErrorString(err));
    return mismatches == 0 && err == hipSuccess;
}

// Multi-wave race repro. We do NOT treat mismatches as a host failure
// because the kernel is intentionally undefined behaviour; the goal is
// to demonstrate that gfx9/gfx950 happens to make it work (and to
// document where the latent bug is).
static bool run_partB_multi_wave()
{
    constexpr int N = kPartBMultiWaveSize;
    int* d_in       = nullptr;
    int* d_out      = nullptr;
    std::vector<int> h_in(N);
    for(int i = 0; i < N; ++i)
        h_in[i] = i * 10;

    (void)hipMalloc(&d_in, N * sizeof(int));
    (void)hipMalloc(&d_out, N * sizeof(int));
    (void)hipMemcpy(d_in, h_in.data(), N * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0xCD, N * sizeof(int));

    hipLaunchKernelGGL(partB_direct_load_shape_multi_wave_kernel,
                       dim3(1),
                       dim3(N),
                       0,
                       nullptr,
                       d_in,
                       d_out);
    auto err = hipDeviceSynchronize();

    std::vector<int> h_out(N);
    (void)hipMemcpy(h_out.data(), d_out, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_in);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < N; ++i)
    {
        const int expected = ((i + 64) % N) * 10;
        if(h_out[i] != expected)
            ++mismatches;
    }

    printf("[B %-26s] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           "direct_load (multi-wave)",
           h_out[0],
           h_out[1],
           h_out[2],
           h_out[3],
           mismatches,
           N,
           hipGetErrorString(err));
    if(mismatches == 0)
        printf("    NOTE: passed despite the buggy wrapper. On gfx9/gfx950 the\n"
               "          s_barrier de facto stalls long enough for in-flight\n"
               "          ds_writes to commit. NOT guaranteed by the ISA.\n");
    else
        printf("    NOTE: race materialized (%d/%d mismatches). This is what\n"
               "          using the wrong wrapper actually buys you.\n",
               mismatches,
               N);
    return err == hipSuccess; // tolerate mismatches: the kernel is UB on purpose
}

// Async buffer_load lds + block_sync_lds_direct_load: the real,
// correct usage. Should always pass.
static bool run_partB_async_load()
{
    constexpr int N = kPartBAsyncSize;
    int* d_in       = nullptr;
    int* d_out      = nullptr;
    std::vector<int> h_in(N);
    for(int i = 0; i < N; ++i)
        h_in[i] = i * 10;

    (void)hipMalloc(&d_in, N * sizeof(int));
    (void)hipMalloc(&d_out, N * sizeof(int));
    (void)hipMemcpy(d_in, h_in.data(), N * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0xCD, N * sizeof(int));

    hipLaunchKernelGGL(partB_async_buffer_load_lds_kernel,
                       dim3(1),
                       dim3(N),
                       0,
                       nullptr,
                       d_in,
                       d_out);
    auto err = hipDeviceSynchronize();

    std::vector<int> h_out(N);
    (void)hipMemcpy(h_out.data(), d_out, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_in);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < N; ++i)
    {
        const int expected = ((i + 1) % N) * 10;
        if(h_out[i] != expected)
            ++mismatches;
    }

    printf("[B %-26s] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           "async buffer_load lds",
           h_out[0],
           h_out[1],
           h_out[2],
           h_out[3],
           mismatches,
           N,
           hipGetErrorString(err));
    return mismatches == 0 && err == hipSuccess;
}

static bool run_partC()
{
    uint32_t* d_counter = nullptr;
    int* d_payload      = nullptr;
    int* d_out          = nullptr;

    (void)hipMalloc(&d_counter, sizeof(uint32_t));
    (void)hipMalloc(&d_payload, kPartCThreads * sizeof(int));
    (void)hipMalloc(&d_out, kPartCThreads * sizeof(int));
    (void)hipMemset(d_counter, 0, sizeof(uint32_t));
    (void)hipMemset(d_out, 0xCD, kPartCThreads * sizeof(int));

    // Two blocks. Block 0 produces, block 1 consumes.
    hipLaunchKernelGGL(partC_kernel,
                       dim3(2),
                       dim3(kPartCThreads),
                       0,
                       nullptr,
                       d_counter,
                       d_payload,
                       d_out);
    auto err = hipDeviceSynchronize();

    std::vector<int> h_out(kPartCThreads);
    (void)hipMemcpy(h_out.data(), d_out, kPartCThreads * sizeof(int), hipMemcpyDeviceToHost);

    (void)hipFree(d_counter);
    (void)hipFree(d_payload);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < kPartCThreads; ++i)
    {
        if(h_out[i] != i * 7)
            ++mismatches;
    }
    printf("[C workgroup_barrier     ] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           h_out[0],
           h_out[1],
           h_out[2],
           h_out[3],
           mismatches,
           kPartCThreads,
           hipGetErrorString(err));
    return mismatches == 0 && err == hipSuccess;
}

// --------------------------------------------------------------------------
// Closing notes: wrappers we do NOT runtime-demo
//
//   async_buffer_load_fence(cnt)
//     defined at amd_buffer_addressing_builtins.hpp:1317
//       asm volatile("s_waitcnt vmcnt(%0)" :: "n"(cnt) : "memory");
//     A vmcnt-only wait with NO barrier. Used right after issuing async
//     "buffer_load ... lds" on the producer side, when you don't yet
//     need to sync the workgroup but you do need to know the load has
//     landed (e.g. before re-using the LDS region for the next stage).
//
//   s_nop(cnt)
//     defined at arch.hpp:1072-1079
//       asm volatile("s_nop %0" :: "n"(cnt) :);
//     Inserts cnt empty cycles. Sometimes used inside hand-tuned
//     pipelines (e.g. before the "lds" form of buffer_load on certain
//     archs to satisfy LDS-write hazards). Not a synchronizer in the
//     usual sense, but it lives in the same toolkit.
//
//   __builtin_amdgcn_s_sleep(n)
//     emits "s_sleep n", which suspends the wave for ~64*(n+1) cycles
//     and lets other waves on the CU issue. Used INSIDE
//     workgroup_barrier::wait_eq_wave (workgroup_barrier.hpp:47) so
//     the spinner doesn't burn issue slots that the producer might
//     need.
//
//   m0_set_with_memory(v) / m0_inc_with_memory(v)
//     defined at utility.hpp:19-28
//       asm volatile("s_mov_b32 m0, %0" :: "s"(v) : "memory");
//       asm volatile("s_add_u32 m0, %0, m0" :: "n"(v) : "memory");
//     Programs the m0 sgpr that a "buffer_load ... lds" instruction
//     uses as the LDS-side base address. The "memory" clobber stops
//     the compiler from reordering pending memory ops across the m0
//     write, which is why these helpers exist as wrappers rather than
//     as plain asm. Demonstrated above in
//     partB_async_buffer_load_lds_kernel; tile_window.hpp:524 and
//     tile_scatter_gather.hpp:652 are the production callers.
//
//   __builtin_amdgcn_iglp_opt(N)
//     "Interleaved Global / LDS Prefetch" hint. An ALTERNATIVE to
//     placing sched_barrier(0) by hand: opt the kernel into one of
//     LLVM's canned schedules. Used by
//     ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp.
//     If iglp_opt picks a good schedule for your kernel, you can throw
//     away most of your sched_barrier(0) calls. If it doesn't, you
//     stick with sched_barrier and place the boundaries manually as
//     unified_attention_pipeline.hpp does.
// --------------------------------------------------------------------------

int main()
{
    printf("=== Tutorial 14.17: sync wrappers ===\n\n");

    printf("-- PART A: block_sync_lds<0>() == s_waitcnt_lgkmcnt<0> + s_barrier --\n");
    bool okA = run_partA();

    printf("\n-- PART B: block_sync_lds[_direct_load]<0>() == line-1054 idiom --\n");
    bool okB1 = run_partB(/*use_direct_load_shape=*/false);
    bool okB2 = run_partB(/*use_direct_load_shape=*/true);
    bool okB3 = run_partB_multi_wave();
    bool okB4 = run_partB_async_load();

    printf("\n-- PART C: workgroup_barrier::wait_eq_wave (cross-WORKGROUP) --\n");
    bool okC = run_partC();

    printf("\n=== Result: A=%s  B(wrapped)=%s  B(direct_load)=%s\n"
           "             B(multi-wave UB)=%s  B(async)=%s  C=%s ===\n",
           okA ? "PASS" : "FAIL",
           okB1 ? "PASS" : "FAIL",
           okB2 ? "PASS" : "FAIL",
           okB3 ? "ran" : "fail",
           okB4 ? "PASS" : "FAIL",
           okC ? "PASS" : "FAIL");

    return (okA && okB1 && okB2 && okB3 && okB4 && okC) ? 0 : 1;
}
