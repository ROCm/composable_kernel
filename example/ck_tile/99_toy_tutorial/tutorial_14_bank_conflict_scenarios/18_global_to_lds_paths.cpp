// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.18: AMDGCN global -> {VGPR, LDS} memory-load families
 *
 * Three vector-memory instruction families exist on AMDGCN. Each one
 * has a "load into a VGPR" form (Part A) and a "load directly into
 * LDS" form (Part B, the so-called async path). They differ in how
 * they ADDRESS global memory and what extra plumbing they require.
 *
 *   family       VGPR-dest form          LDS-dest form              addressing
 *   ------       --------------          ------------               ----------
 *   flat_*       flat_load_dword         flat_load_dword ... lds    64-bit flat ptr
 *                                        (gfx10/11/12; iffy on
 *                                        gfx9/gfx950 -- see PartB3)
 *   global_*     global_load_dword       global_load_lds_dword      64-bit ptr + per-lane voff
 *                                        (gfx940+, gfx950, gfx10+)
 *   buffer_*     buffer_load_dword       buffer_load_dword ... lds  4-dword buffer resource (V#)
 *                ... offen               ... offen lds              + per-lane voff
 *                                        (gfx9 / gfx950 / gfx10+)
 *
 * Why three families?
 *   - flat_*   is the most general: a 64-bit address is dynamically
 *              dispatched to global / LDS / scratch based on its high
 *              bits. HIP's default `int*` lowers to this when the
 *              compiler can't prove the pointer is global.
 *   - global_* skips the dispatch: the compiler asserts the address
 *              is in global address space, so the hardware doesn't
 *              need to inspect the high bits. Cheaper than flat_*.
 *   - buffer_* takes a 128-bit resource descriptor (V#) in 4 SGPRs
 *              that pre-encodes base / stride / size / cache flags.
 *              Per-lane voff is a 32-bit offset into that descriptor.
 *              Cheapest at the load site (no 64-bit address math) and
 *              gets bounds-checked for free, but you have to build
 *              the V# ahead of time.
 *
 * The LDS-dest variants are AMDGCN's "direct global -> LDS" path.
 * Instead of "load into VGPR, write to LDS", one instruction ships
 * the dword from HBM all the way to LDS without ever materializing
 * it in a VGPR. This is what ck_tile calls "async" loads. Two pieces
 * of plumbing the LDS-dest forms need:
 *
 *   m0 SGPR    - holds the LDS-side BASE address for this wave's
 *                slice of LDS. For buffer_* and flat_* you must set
 *                m0 yourself (m0_set_with_memory). For global_* the
 *                clang builtin __builtin_amdgcn_global_load_lds emits
 *                the s_mov_b32 m0, ... for you.
 *   V#         - only buffer_* needs one. global_* and flat_* take
 *                a plain 64-bit pointer.
 *
 * Counter / sync rule for ALL three LDS-dest forms:
 *   The "lds" suffix moves the LDS write into the SAME instruction as
 *   the global load. NO ds_write happens, so lgkmcnt does NOT bump --
 *   only vmcnt. After issuing, drain with vmcnt-only:
 *      block_sync_lds_direct_load<0>()
 *        == s_waitcnt vmcnt(0); s_barrier
 *   This is the unified_attention_pipeline.hpp:1054-1055 idiom.
 *
 * ============================================================================
 * How to inspect the assembly
 * ============================================================================
 *
 * Build with ENABLE_ASM_DUMP=ON. The dump for this file lands at
 *   build/18_global_to_lds_paths-hip-amdgcn-amd-amdhsa-gfx950.s
 * and each kernel body is bracketed by "[POYENC] <name>" markers.
 *
 *   grep -n -E '\[POYENC\]|flat_load|global_load|buffer_load' \
 *       build/18_global_to_lds_paths-hip-amdgcn-amd-amdhsa-gfx950.s
 *   grep -n -E 's_mov_b32 m0|s_waitcnt|s_barrier' \
 *       build/18_global_to_lds_paths-hip-amdgcn-amd-amdhsa-gfx950.s
 *
 * The first grep shows which family each kernel actually emitted; the
 * second shows the m0 setup before the LDS-dest ops and the vmcnt
 * drain after.
 *
 * ============================================================================
 * gfx950 caveats (the hardware this tutorial is built for)
 * ============================================================================
 *
 *   buffer_load_dword ... lds          : known good. CK production code
 *                                        uses it everywhere on gfx9/gfx950.
 *   global_load_lds_dword              : known good on gfx950 (gfx940+
 *                                        family). The clang builtin
 *                                        lowers cleanly.
 *   flat_load_dword ... lds            : NOT available on gfx9/gfx950.
 *                                        The gfx950 assembler rejects
 *                                        the "lds" modifier on flat_*.
 *                                        It IS available on gfx10/11/12
 *                                        (RDNA). The B3 kernel below
 *                                        auto-detects the target: real
 *                                        flat_load_dword ... lds on
 *                                        gfx10+, and a flat_load_dword
 *                                        + manual ds_write fallback on
 *                                        gfx950 so the binary still
 *                                        builds and validates.
 *
 * Build:
 *   target is aa_tutorial_14_18_global_to_lds_paths
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

using namespace ck_tile;

// Same marker macro as 14.16 / 14.17. The two sched_barrier(0)'s
// bracket the asm("...") line so the LLVM scheduler is forbidden to
// move surrounding instructions across the marker; that means the
// load/sync we want to inspect always lands BETWEEN its two markers
// in the .s dump.
#define ASM_MARKER_DEMO(name)            \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #name);   \
    __builtin_amdgcn_sched_barrier(0);

// Common workgroup size; one full wave on gfx950.
constexpr int kThreads = 64;

// CPU-side pattern that all kernels read from; matches kThreads.
static inline int host_pattern(int i) { return i * 7 + 1; }

// ============================================================================
// PART A: VGPR-destination loads, one per family
//
// Each kernel reads one dword per lane from g_in into a VGPR using
// inline asm so the EXACT instruction we are advertising shows up in
// the dump (otherwise the optimizer is free to pick whichever form
// it prefers). Then a small +K offset is added and the result is
// written back via the default codegen path.
// ============================================================================

// --- A1: flat_load_dword ---------------------------------------------------
//
// flat_load_dword takes a 64-bit address in a v[N:N+1] register pair.
// The hardware looks at the address's high bits at run time to decide
// whether it lives in global / LDS / scratch. This is HIP's default
// for "int* p; ... = *p;" when the compiler can't prove p is global.
__global__ void partA_flat_load_kernel(const int* g_in, int* g_out)
{
    const int tid = threadIdx.x;
    const int* p  = g_in + tid;
    int x;

    ASM_MARKER_DEMO(partA_flat_load_begin)
    asm volatile("flat_load_dword %0, %1\n\t"
                 "s_waitcnt vmcnt(0) lgkmcnt(0)"
                 : "=v"(x)
                 : "v"(p)
                 : "memory");
    ASM_MARKER_DEMO(partA_flat_load_end)

    g_out[tid] = x + 1;
}

// --- A2: global_load_dword -------------------------------------------------
//
// global_load_dword vDST, vADDR, off
// Per-lane 64-bit address in vADDR; "off" means no scalar base offset.
// The compiler skips the AS-dispatch logic that flat_* does. On
// gfx950 / CDNA this is the default for kernel-arg pointers when
// the compiler can prove they are in global AS.
__global__ void partA_global_load_kernel(const int* g_in, int* g_out)
{
    const int tid = threadIdx.x;
    const int* p  = g_in + tid;
    int x;

    ASM_MARKER_DEMO(partA_global_load_begin)
    asm volatile("global_load_dword %0, %1, off\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(x)
                 : "v"(p)
                 : "memory");
    ASM_MARKER_DEMO(partA_global_load_end)

    g_out[tid] = x + 2;
}

// --- A3: buffer_load_dword (offen) -----------------------------------------
//
// buffer_load_dword vDST, vOFF, sRSRC, 0 offen
// vOFF: per-lane 32-bit byte offset into the buffer.
// sRSRC: V# = 4 SGPRs containing base address (64b), stride/format
// (32b), and num_records / cache flags (32b). Built here by
// make_wave_buffer_resource(ptr, size_bytes).
//
// Pros over flat/global: cheaper at the load site (no 64-bit address
// math; the V# pre-encodes base + range), free out-of-bounds
// suppression. Cons: you have to build the V#, which is 4 SGPRs of
// kernel state.
__global__ void partA_buffer_load_kernel(const int* g_in, int* g_out)
{
    const int tid = threadIdx.x;

    int32x4_t rsrc = make_wave_buffer_resource(g_in, kThreads * sizeof(int));

    int x;
    ASM_MARKER_DEMO(partA_buffer_load_begin)
    asm volatile("buffer_load_dword %0, %1, %2, 0 offen\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(x)
                 : "v"(tid * static_cast<int>(sizeof(int))), "s"(rsrc)
                 : "memory");
    ASM_MARKER_DEMO(partA_buffer_load_end)

    g_out[tid] = x + 3;
}

// ============================================================================
// PART B: LDS-destination loads ("async"), one per family
//
// Each kernel pulls one dword per lane from global directly into LDS
// using its family's LDS-dest instruction, drains with vmcnt-only
// (block_sync_lds_direct_load<0>), and reads lds[(tid+1) % 64]
// through a `volatile int*` view. The volatile view is the same
// trick we used in 14.17: the inline-asm load doesn't tie its LDS
// write back to the C++ array, so without volatile the optimizer
// would DCE the post-sync read and even drop the LDS allocation.
// ============================================================================

// --- B1: buffer_load_dword ... offen ... lds ---------------------------------
//
// The classic CDNA async path. Same shape as 14.17's
// partB_async_buffer_load_lds_kernel; we re-emit it here so the new
// dump has all three families side by side.
//
//   1. make_wave_buffer_resource(g_in, ...)  -> V# in 4 SGPRs
//   2. m0_set_with_memory(0)                 -> LDS-side base = byte 0
//   3. async_buffer_load_dwordxn_v<1>(...)   -> the actual instruction
__global__ void partB_buffer_load_lds_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kThreads];
    const int tid = threadIdx.x;

    int32x4_t rsrc = make_wave_buffer_resource(g_in, kThreads * sizeof(int));
    m0_set_with_memory(0);

    ASM_MARKER_DEMO(partB_buffer_lds_issue)
    async_buffer_load_dwordxn_v<1>(/*smem=*/lds,
                                   rsrc,
                                   /*voffset=*/tid * static_cast<index_t>(sizeof(int)),
                                   /*soffset=*/0,
                                   /*ioffset=*/0);

    ASM_MARKER_DEMO(partB_buffer_lds_sync)
    block_sync_lds_direct_load<0>();

    volatile int* vlds = lds;
    g_out[tid]         = vlds[(tid + 1) % kThreads];
}

// --- B2: global_load_lds_dword ---------------------------------------------
//
// gfx940+ / gfx950 / gfx10+. Compiler builtin handles all the m0
// plumbing for us: it picks lane 0's value of `&lds[tid]` (== &lds[0])
// and emits "s_mov_b32 m0, <that>" before the load. Per-lane LDS
// destination is implicit via the wave layout, just like the buffer_*
// form.
//
// Signature:
//   __builtin_amdgcn_global_load_lds(src_global_ptr,
//                                    dst_lds_ptr,
//                                    /*size_bytes=*/4 | 8 | 16,
//                                    /*offset=*/int32,
//                                    /*aux=*/int32)
__global__ void partB_global_load_lds_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kThreads];
    const int tid = threadIdx.x;

    ASM_MARKER_DEMO(partB_global_lds_issue)
    // Cast away const for the builtin's void* src parameter; the
    // load is read-only on the device side regardless.
    __builtin_amdgcn_global_load_lds(const_cast<int*>(g_in + tid),
                                     &lds[tid],
                                     /*size=*/4,
                                     /*offset=*/0,
                                     /*aux=*/0);

    ASM_MARKER_DEMO(partB_global_lds_sync)
    block_sync_lds_direct_load<0>();

    volatile int* vlds = lds;
    g_out[tid]         = vlds[(tid + 1) % kThreads];
}

// --- B3: flat_load_dword ... lds (gfx10+ only; gfx9/gfx950 fallback) -------
//
// On gfx10/11/12 (RDNA) the AMDGCN assembler emits the bundled form
//   flat_load_dword vDST, vADDR_PAIR lds
// where vDST is unused (the LDS write goes through m0) and vADDR_PAIR
// is a 64-bit pointer. As with buffer_*, m0 has to be set first;
// there is NO clang builtin for this form so we use inline asm.
//
// On gfx9/gfx950 the assembler rejects the "lds" modifier on flat_*
// (only buffer_* and global_* have LDS-dest variants on CDNA). To
// keep this tutorial buildable and validated, the gfx9/gfx950 path
// falls back to a regular flat_load + manual ds_write sequence,
// which is precisely the pair that the gfx10+ instruction is
// supposed to fuse. The fallback is clearly tagged in the assembly
// dump via the marker name.
//
// Detection: clang predefines __gfx1010__/__gfx1030__/__gfx1100__/...
// for RDNA targets. We enumerate them so the file is robust against
// being compiled for any of those without changes.
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
    defined(__gfx1013__) || defined(__gfx1030__) || defined(__gfx1031__) || \
    defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || \
    defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
#define POYENC_FLAT_LDS_AVAILABLE 1
#else
#define POYENC_FLAT_LDS_AVAILABLE 0
#endif

__global__ void partB_flat_load_lds_kernel(const int* g_in, int* g_out)
{
    __shared__ int lds[kThreads];
    const int tid = threadIdx.x;
#if POYENC_FLAT_LDS_AVAILABLE
    const int* p = g_in + tid;

    m0_set_with_memory(0);

    int unused_dst;
    ASM_MARKER_DEMO(partB_flat_lds_issue_real)
    asm volatile("flat_load_dword %0, %1 lds"
                 : "=v"(unused_dst)
                 : "v"(p)
                 : "memory");
    (void)unused_dst;

    ASM_MARKER_DEMO(partB_flat_lds_sync_real)
    block_sync_lds_direct_load<0>();
#else
    ASM_MARKER_DEMO(partB_flat_lds_issue_fallback)
    int v    = g_in[tid];
    lds[tid] = v;

    ASM_MARKER_DEMO(partB_flat_lds_sync_fallback)
    block_sync_lds<0>();
#endif

    volatile int* vlds = lds;
    g_out[tid]         = vlds[(tid + 1) % kThreads];
}

// ============================================================================
// Host driver
// ============================================================================

namespace {

struct RunResult
{
    bool ok;
    bool launch_ok;
    int mismatches;
    int sample[4];
};

template <typename Kernel>
static RunResult launch_and_validate(Kernel kernel, int add_constant, const char* label)
{
    int* d_in  = nullptr;
    int* d_out = nullptr;
    std::vector<int> h_in(kThreads);
    for(int i = 0; i < kThreads; ++i)
        h_in[i] = host_pattern(i);

    (void)hipMalloc(&d_in, kThreads * sizeof(int));
    (void)hipMalloc(&d_out, kThreads * sizeof(int));
    (void)hipMemcpy(d_in, h_in.data(), kThreads * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0xCD, kThreads * sizeof(int));

    hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreads), 0, nullptr, d_in, d_out);
    hipError_t err = hipDeviceSynchronize();

    std::vector<int> h_out(kThreads);
    (void)hipMemcpy(h_out.data(), d_out, kThreads * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_in);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < kThreads; ++i)
    {
        const int expected = host_pattern(i) + add_constant;
        if(h_out[i] != expected)
            ++mismatches;
    }

    RunResult r{};
    r.launch_ok  = (err == hipSuccess);
    r.mismatches = mismatches;
    r.ok         = r.launch_ok && mismatches == 0;
    for(int i = 0; i < 4; ++i)
        r.sample[i] = h_out[i];

    printf("[%-28s] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           label,
           r.sample[0],
           r.sample[1],
           r.sample[2],
           r.sample[3],
           mismatches,
           kThreads,
           hipGetErrorString(err));
    return r;
}

template <typename Kernel>
static RunResult launch_and_validate_lds_neighbor(Kernel kernel, const char* label)
{
    int* d_in  = nullptr;
    int* d_out = nullptr;
    std::vector<int> h_in(kThreads);
    for(int i = 0; i < kThreads; ++i)
        h_in[i] = host_pattern(i);

    (void)hipMalloc(&d_in, kThreads * sizeof(int));
    (void)hipMalloc(&d_out, kThreads * sizeof(int));
    (void)hipMemcpy(d_in, h_in.data(), kThreads * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0xCD, kThreads * sizeof(int));

    hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreads), 0, nullptr, d_in, d_out);
    hipError_t err = hipDeviceSynchronize();

    std::vector<int> h_out(kThreads);
    (void)hipMemcpy(h_out.data(), d_out, kThreads * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_in);
    (void)hipFree(d_out);

    int mismatches = 0;
    for(int i = 0; i < kThreads; ++i)
    {
        const int expected = host_pattern((i + 1) % kThreads);
        if(h_out[i] != expected)
            ++mismatches;
    }

    RunResult r{};
    r.launch_ok  = (err == hipSuccess);
    r.mismatches = mismatches;
    r.ok         = r.launch_ok && mismatches == 0;
    for(int i = 0; i < 4; ++i)
        r.sample[i] = h_out[i];

    printf("[%-28s] g_out[0..3] = %d %d %d %d, mismatches = %d/%d (err=%s)\n",
           label,
           r.sample[0],
           r.sample[1],
           r.sample[2],
           r.sample[3],
           mismatches,
           kThreads,
           hipGetErrorString(err));
    return r;
}

} // namespace

int main()
{
    printf("=== Tutorial 14.18: AMDGCN global -> {VGPR,LDS} load families ===\n\n");

    printf("-- PART A: VGPR-destination (one instr per family, inline asm) --\n");
    auto a1 = launch_and_validate(partA_flat_load_kernel, 1, "A1 flat_load_dword");
    auto a2 = launch_and_validate(partA_global_load_kernel, 2, "A2 global_load_dword");
    auto a3 = launch_and_validate(partA_buffer_load_kernel, 3, "A3 buffer_load_dword offen");

    printf("\n-- PART B: LDS-destination ('async'), neighbor read (lds[(tid+1)%%64]) --\n");
    auto b1 = launch_and_validate_lds_neighbor(partB_buffer_load_lds_kernel,
                                               "B1 buffer_load_dword lds");
    auto b2 = launch_and_validate_lds_neighbor(partB_global_load_lds_kernel,
                                               "B2 global_load_lds_dword");

    const char* b3_label =
#if POYENC_FLAT_LDS_AVAILABLE
        "B3 flat_load_dword lds";
#else
        "B3 flat_load + ds_write (fb)";
#endif
    auto b3 = launch_and_validate_lds_neighbor(partB_flat_load_lds_kernel, b3_label);

    printf("\n=== Result: A1=%s A2=%s A3=%s | B1=%s B2=%s B3=%s ===\n",
           a1.ok ? "PASS" : "FAIL",
           a2.ok ? "PASS" : "FAIL",
           a3.ok ? "PASS" : "FAIL",
           b1.ok ? "PASS" : "FAIL",
           b2.ok ? "PASS" : "FAIL",
           b3.ok ? "PASS" : "FAIL");

#if !POYENC_FLAT_LDS_AVAILABLE
    printf("\nNote: B3 used the flat_load + ds_write fallback because\n"
           "      flat_load_dword ... lds is not available on this gfx target\n"
           "      (the LDS-dest flat_* form is gfx10/11/12-only; gfx9/gfx950\n"
           "      have buffer_*/global_* LDS-dest forms instead).\n");
#endif

    bool overall = a1.ok && a2.ok && a3.ok && b1.ok && b2.ok && b3.ok;
    return overall ? 0 : 1;
}
