// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.16: AMDGCN synchronization primitives, side by side
 *
 * The companion file is the unified attention pipeline at
 *   include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
 * Lines 1054-1055 of that file are the single most-quoted idiom in the
 * project:
 *
 *     s_waitcnt_vmcnt<0>();
 *     __builtin_amdgcn_s_barrier();
 *
 * This tutorial unpacks every primitive that appears in those two lines
 * and the surrounding pipeline, with the smallest possible runnable
 * examples for each. Once you've seen them in isolation, the pipeline
 * stops looking like magic.
 *
 *   PART A - s_barrier + s_waitcnt_lgkmcnt:
 *            LDS producer/consumer, the cleanest case where you have to
 *            wait on a memory counter (lgkmcnt) AND order all the waves
 *            of the block (s_barrier). Three variants: race, half-fixed,
 *            correct.
 *
 *   PART B - s_waitcnt_vmcnt + s_barrier (the line-1054 idiom):
 *            global -> VGPR -> LDS staging followed by a cross-thread
 *            LDS read. Shows the same pattern as Demo A but on the VMEM
 *            side of the counter pair.
 *
 *   PART C - ASM_MARKER and sched_barrier(0):
 *            tiny "do nothing useful" kernels designed to be inspected
 *            in assembly. See the closing comment block for the exact
 *            hipcc -S command and what to look for. Also shows the
 *            contrast with __builtin_amdgcn_s_setprio which DOES emit
 *            an instruction (unlike sched_barrier, which does not).
 *
 * ============================================================================
 * Quick reference: the three per-wave hardware counters
 * ============================================================================
 *
 *   vmcnt   - outstanding vector-memory ops:
 *               buffer_load/store_*, global_load/store_*, flat_load/store_*,
 *               and async "buffer_load ... lds" (the bundled global->LDS path).
 *             Wait on this BEFORE you read a destination that a global load
 *             targeted.
 *
 *   lgkmcnt - outstanding LDS / GDS / scalar-memory ops:
 *               ds_read/write_*, s_load_*. The mnemonic spells out
 *               LDS + GDS + sKalar + sMem.
 *             Wait on this BEFORE you consume the result of a ds_read or
 *             rely on a ds_write being visible to a s_barrier.
 *
 *   expcnt  - outstanding export ops (color/position output for graphics).
 *             Compute kernels almost never wait on it; ck_tile passes the
 *             max value to mean "do not wait".
 *
 * The bit-layout of the s_waitcnt mask differs per architecture; ck_tile
 * encodes all three layouts in arch.hpp:908-953. The two helpers used by
 * the unified attention pipeline,
 *
 *   s_waitcnt_vmcnt<N>()    = wait on VMEM, ignore LDS
 *   s_waitcnt_lgkmcnt<N>()  = wait on LDS,  ignore VMEM
 *
 * are defined inline at the top of unified_attention_pipeline.hpp:142-162
 * and hard-code the legacy gfx9 bit-layout. We replicate them here as
 * static device functions so the demos read identically to the pipeline.
 *
 * ============================================================================
 * The five primitives covered in this file
 * ============================================================================
 *
 *   __builtin_amdgcn_s_barrier()
 *      SOPP "s_barrier". Workgroup-wide execution barrier. All waves of
 *      the block must arrive before any wave proceeds. NOT a memory
 *      fence: pair it with an s_waitcnt on the producer side.
 *
 *   __builtin_amdgcn_s_waitcnt(mask)   (and our s_waitcnt_vmcnt / lgkmcnt wrappers)
 *      SOPP "s_waitcnt". Stalls THIS wave only, until each counter
 *      encoded in mask has dropped to <= the encoded value.
 *
 *   __builtin_amdgcn_s_setprio(p)
 *      SOPP "s_setprio". Sets this wave's scheduler priority on the CU
 *      to p in [0..3]. The unified attention pipeline uses this at lines
 *      1135 / 1142 so the compute warp-group out-issues the prefetch
 *      warp-group. NOT a barrier and NOT a counter wait.
 *
 *   __builtin_amdgcn_sched_barrier(mask)
 *      Compiler-only hint, emits ZERO machine instructions. Mask=0
 *      forbids the LLVM scheduler from moving any instruction across
 *      this point. Used everywhere in the pipeline (lines 13/15, 848+)
 *      to lock the hand-tuned interleave.
 *
 *   ASM_MARKER(name) macro
 *      = sched_barrier(0); asm("; [POYENC] name"); sched_barrier(0).
 *      Inserts a single comment line (no opcode) into the AMDGCN dump
 *      that the scheduler is forbidden to move past. Lets you grep the
 *      ISA dump and locate exactly which source phase compiled to which
 *      block of instructions. Lives at unified_attention_pipeline.hpp:12-18.
 *
 * Build:
 *   target is aa_tutorial_14_16_sync_primitives_intro
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

using namespace ck_tile;

// --------------------------------------------------------------------------
// Local helpers replicating unified_attention_pipeline.hpp:142-162.
// We reproduce them verbatim so the demo code reads exactly like the
// pipeline. These hard-code the legacy gfx9 waitcnt bit-layout
// (vm = bits [15:14]|[3:0], lgkm = bits [11:8], exp = bits [6:4]).
// --------------------------------------------------------------------------

template <uint16_t Vmcnt, uint8_t Lgkmcnt, uint8_t Expcnt = 7>
__device__ static constexpr void s_waitcnt_local()
{
    __builtin_amdgcn_s_waitcnt((((0b110000 & Vmcnt) << (14 - 4)) | (0b1111 & Vmcnt)) |
                               ((0b111 & Expcnt) << 4) | ((0b1111 & Lgkmcnt) << 8));
}

template <uint16_t Vmcnt>
__device__ static constexpr void s_waitcnt_vmcnt_local()
{
    s_waitcnt_local<Vmcnt, 15>();
}

template <uint8_t Lgkmcnt>
__device__ static constexpr void s_waitcnt_lgkmcnt_local()
{
    s_waitcnt_local<63, Lgkmcnt>();
}
//

#define ASM_MARKER_DEMO(name)            \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #name);   \
    __builtin_amdgcn_sched_barrier(0);

// --------------------------------------------------------------------------
// PART A: __builtin_amdgcn_s_barrier + s_waitcnt_lgkmcnt
//
// The minimal LDS producer/consumer. Thread 0 stores 42 into a one-slot
// LDS array; all 64 threads read it back. There are three variants
// chosen to expose what each primitive does in isolation.
//
//   A0 - "no_sync"     : nothing between the write and the read.
//                        Threads 1..63 race against thread 0's ds_write
//                        and may observe whatever was in LDS at kernel
//                        entry (typically zero on a fresh launch).
//   A1 - "barrier_only": __builtin_amdgcn_s_barrier() between write and
//                        read. Workgroup execution is now ordered (the
//                        barrier waits for all waves to arrive), but
//                        thread 0's ds_write may still be in flight at
//                        the barrier point on archs where s_barrier
//                        does not auto-wait on lgkmcnt -- so the read
//                        side may still see the old value.
//   A2 - "correct"     : s_waitcnt_lgkmcnt<0>(); __builtin_amdgcn_s_barrier();
//                        on the producer side. Memory ordering AND wave
//                        ordering. This is what block_sync_lds<0>()
//                        compiles to (see arch.hpp:1060-1064).
//
// On gfx9/10/11 the "barrier_only" variant frequently happens to work
// because s_barrier in practice waits long enough for the in-flight
// ds_write, but that is not guaranteed by the ISA. The lesson:
//   s_barrier orders WAVES; s_waitcnt orders MEMORY within a wave.
//   You almost always need both.
// --------------------------------------------------------------------------

enum class SyncMode
{
    no_sync,
    barrier_only,
    correct,
};

template <SyncMode Mode>
__global__ void partA_kernel(int* output)
{
    __shared__ int lds[1];
    const int tid = threadIdx.x;

    if(tid == 0)
    {
        // ds_write -> bumps lgkmcnt. The producer wave now has one
        // outstanding LDS-write op until lgkmcnt drops back to 0.
        lds[0] = 42;

        if constexpr(Mode == SyncMode::correct)
        {
            // Drain my own outstanding ds_write before letting other
            // waves observe the LDS slot via the barrier below.
            s_waitcnt_lgkmcnt_local<0>();
        }
    }

    if constexpr(Mode == SyncMode::barrier_only || Mode == SyncMode::correct)
    {
        // Workgroup execution barrier. Every wave of the block must
        // reach this point before any wave proceeds.
        __builtin_amdgcn_s_barrier();
    }
    // SyncMode::no_sync falls straight through with no synchronization.

    // ds_read -> bumps lgkmcnt. The compiler will auto-insert a
    // s_waitcnt lgkmcnt(0) before the assignment to output[tid] below
    // because the dependency on `value` is visible.
    int value      = lds[0];
    output[tid]    = value;
}

// --------------------------------------------------------------------------
// PART B: s_waitcnt_vmcnt<0>(); __builtin_amdgcn_s_barrier();
//
// The line-1054 idiom in its natural habitat: stage a tile from global
// memory through LDS, then read it back across thread lanes.
//
//   1. Each thread issues a global load g_in[tid] into a VGPR.
//      That bumps vmcnt.
//   2. We wait for vmcnt to drain so the VGPR holds the loaded value.
//   3. Each thread writes the VGPR into lds[tid]. That bumps lgkmcnt.
//   4. We wait for lgkmcnt to drain, then s_barrier so that EVERY
//      thread's lds[tid] is visible to every other thread.
//      The combo "s_waitcnt_lgkmcnt<0>(); __builtin_amdgcn_s_barrier();"
//      is what block_sync_lds<0>() (arch.hpp:1060-1064) does.
//   5. Each thread reads lds[(tid+1) % 64] -- a CROSS-THREAD read.
//      Without step 4, this would race with the producer's ds_write
//      from a different lane.
//   6. Wait for the ds_read to land in a VGPR before storing it back
//      to global. (This particular wait is also auto-inserted by the
//      compiler because g_out[tid] = neighbor depends on it.)
//
// In the unified attention pipeline this exact pattern is at
//   ...:1054   s_waitcnt_vmcnt<0>();           <-- after async global->LDS
//   ...:1055   __builtin_amdgcn_s_barrier();   <-- everyone agrees
//   ...:1059   V_lds_load(number<0>{});        <-- ds_read into kv_tile.v_tile
//   ...:1060   s_waitcnt_lgkmcnt<0>();         <-- before gemm consumes it
// The pipeline fires the global->LDS load via the async "buffer_load lds"
// path, which bundles the LDS write into the global op and only bumps
// vmcnt (not lgkmcnt) on the LDS side. That is why line 1054 waits on
// vmcnt and not on lgkmcnt -- the LDS write counter never moved.
//
// We use the simpler global->VGPR->LDS roundtrip here so the demo runs
// on any gfx, but the SHAPE of the code is the same.
// --------------------------------------------------------------------------

constexpr int kPartBSize = 64;

__global__ void partB_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kPartBSize];
    const int tid = threadIdx.x;

    // Step 1: global load, bumps vmcnt.
    int v = g_in[tid];

    // Step 2: drain vmcnt.
    //         (The compiler would insert this anyway here because v is
    //          consumed by the next line; we spell it out so the call
    //          site mirrors unified_attention_pipeline.hpp:1054.)
    s_waitcnt_vmcnt_local<0>();

    // Step 3: stage into LDS, bumps lgkmcnt.
    lds[tid] = v;

    // Step 4: drain lgkmcnt + workgroup barrier. This is the same
    //         pattern as line 1054-1055 of the pipeline, just on the
    //         lgkmcnt side instead of the vmcnt side.
    s_waitcnt_lgkmcnt_local<0>();
    __builtin_amdgcn_s_barrier();

    // Step 5: cross-thread LDS read. Lane tid reads what lane (tid+1)
    //         wrote. Without step 4 this would race.
    int neighbor = lds[(tid + 1) % kPartBSize];

    // Step 6: drain the ds_read counter before consuming the result.
    s_waitcnt_lgkmcnt_local<0>();

    g_out[tid] = neighbor;
}

// --------------------------------------------------------------------------
// PART C: ASM_MARKER and __builtin_amdgcn_sched_barrier
//
// These two primitives have NO observable runtime effect on this kernel,
// so the "demo" is to compile the file and grep the AMDGCN assembly.
// The kernel does a deliberately cheap arithmetic chain so the dump is
// short and readable.
// --------------------------------------------------------------------------

__global__ void partC_marker_kernel(int* out, int x, int y)
{
    ASM_MARKER_DEMO(phase_alpha)
    int a = x * 3 + 7;
    int b = a ^ 0x55;
    ASM_MARKER_DEMO(phase_beta)
    int c = b + y * y;
    *out  = c;
}

// Contrast kernel: __builtin_amdgcn_s_setprio DOES emit a real
// "s_setprio" SOPP instruction. Useful proof that some "compiler"
// builtins are actually scheduler hints (sched_barrier) while others
// are real ISA (s_setprio, s_barrier, s_waitcnt).
__global__ void partC_setprio_kernel(int* out, int x)
{
    __builtin_amdgcn_s_setprio(1);
    int v = x + 1;
    __builtin_amdgcn_s_setprio(0);
    *out = v;
}

// --------------------------------------------------------------------------
// Host driver
// --------------------------------------------------------------------------

template <SyncMode Mode>
static void run_partA(const char* label)
{
    constexpr int kThreads = 64;
    int* d_out             = nullptr;
    (void)hipMalloc(&d_out, kThreads * sizeof(int));
    (void)hipMemset(d_out, 0xCD, kThreads * sizeof(int));

    hipLaunchKernelGGL(partA_kernel<Mode>, dim3(1), dim3(kThreads), 0, nullptr, d_out);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "[A %-13s] kernel launch failed: %s\n", label, hipGetErrorString(err));
        (void)hipFree(d_out);
        return;
    }

    std::vector<int> h_out(kThreads);
    (void)hipMemcpy(h_out.data(), d_out, kThreads * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_out);

    // Look for any thread that did NOT see 42.
    int wrong_threads = 0;
    int first_wrong   = -1;
    for(int i = 0; i < kThreads; ++i)
    {
        if(h_out[i] != 42)
        {
            ++wrong_threads;
            if(first_wrong < 0)
                first_wrong = i;
        }
    }

    printf("[A %-13s] tid 0 = %d, tid 63 = %d, wrong_threads = %d/%d",
           label,
           h_out[0],
           h_out[kThreads - 1],
           wrong_threads,
           kThreads);
    if(first_wrong >= 0)
        printf(" (first wrong at tid %d, value %d)", first_wrong, h_out[first_wrong]);
    printf("\n");
}

static void run_partB()
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

    hipLaunchKernelGGL(partB_kernel, dim3(1), dim3(kPartBSize), 0, nullptr, d_in, d_out);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "[B] kernel launch failed: %s\n", hipGetErrorString(err));
        (void)hipFree(d_in);
        (void)hipFree(d_out);
        return;
    }

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
    printf("[B line-1054 idiom] g_out[0..3] = %d %d %d %d  (expected 10 20 30 40), mismatches = %d/%d\n",
           h_out[0],
           h_out[1],
           h_out[2],
           h_out[3],
           mismatches,
           kPartBSize);
}

static void run_partC()
{
    int* d_out = nullptr;
    (void)hipMalloc(&d_out, sizeof(int));

    hipLaunchKernelGGL(partC_marker_kernel, dim3(1), dim3(1), 0, nullptr, d_out, 5, 7);
    (void)hipDeviceSynchronize();
    int marker_out = -1;
    (void)hipMemcpy(&marker_out, d_out, sizeof(int), hipMemcpyDeviceToHost);

    hipLaunchKernelGGL(partC_setprio_kernel, dim3(1), dim3(1), 0, nullptr, d_out, 100);
    (void)hipDeviceSynchronize();
    int setprio_out = -1;
    (void)hipMemcpy(&setprio_out, d_out, sizeof(int), hipMemcpyDeviceToHost);

    (void)hipFree(d_out);

    // Both kernels run, but the educational payload of PART C is the
    // assembly dump, not these numbers.
    printf("[C marker_kernel ] computed (5*3+7)^0x55 + 7*7 = %d\n", marker_out);
    printf("[C setprio_kernel] computed 100 + 1            = %d\n", setprio_out);
    printf("\n");
    printf("To see the difference between sched_barrier (compiler-only)\n");
    printf("and s_setprio (real ISA), dump the assembly:\n");
    printf("\n");
    printf("  hipcc -S --offload-arch=gfx942 \\\n");
    printf("        -I<repo>/aiter/3rdparty/composable_kernel/include \\\n");
    printf("        -I<repo>/aiter/3rdparty/composable_kernel/example \\\n");
    printf("        16_sync_primitives_intro.cpp -o - \\\n");
    printf("    | grep -nE 'POYENC|s_setprio|s_barrier|sched|s_waitcnt|s_wait_'\n");
    printf("\n");
    printf("Expected observations:\n");
    printf("  * '; [POYENC] phase_alpha' and 'phase_beta' appear in source order.\n");
    printf("  * NO sched_barrier instruction in the dump (compiler-only hint).\n");
    printf("  * The v_mul/v_xor/v_mad for a, b, c are NOT reordered across\n");
    printf("    the markers. (Try removing sched_barrier(0) and recompile.)\n");
    printf("  * partC_setprio_kernel DOES contain two 's_setprio' instructions.\n");
    printf("  * Both partA correct/barrier-only kernels contain real\n");
    printf("    's_barrier' and 's_waitcnt' SOPPs.\n");
}

// --------------------------------------------------------------------------
// Note on __builtin_amdgcn_s_setprio runtime effect
//
// Showing that s_setprio actually changes wave-issue order requires two
// warp groups racing for issue slots on the same CU, which is what
// unified_attention_pipeline.hpp arranges at lines 1130-1146:
//
//     if(warp_group_id == 0) {                  // prefetcher
//         __builtin_amdgcn_s_setprio(0);        // step DOWN
//         __builtin_amdgcn_s_barrier();
//         while(core_loop(number<0>{})) ;
//     }
//     if(warp_group_id != 0) {                  // compute
//         __builtin_amdgcn_s_setprio(1);        // step UP
//         __builtin_amdgcn_s_barrier();
//         while(core_loop(number<1>{})) ;
//     }
//
// Reproducing that here would obscure the "what does the primitive do"
// question with a multi-pipeline scaffolding. The takeaway for an intro:
// s_setprio is NOT a barrier and NOT a counter wait -- it only biases
// the CU scheduler. Use it inside hand-tuned producer/consumer pipelines
// only.
// --------------------------------------------------------------------------

int main()
{
    printf("=== Tutorial 14.16: AMDGCN sync primitives ===\n\n");

    printf("-- PART A: __builtin_amdgcn_s_barrier + s_waitcnt_lgkmcnt --\n");
    run_partA<SyncMode::no_sync>("no_sync");
    run_partA<SyncMode::barrier_only>("barrier_only");
    run_partA<SyncMode::correct>("correct");

    printf("\n-- PART B: s_waitcnt_vmcnt<0>() + __builtin_amdgcn_s_barrier --\n");
    run_partB();

    printf("\n-- PART C: ASM_MARKER + sched_barrier + s_setprio (asm dump) --\n");
    run_partC();

    return 0;
}
