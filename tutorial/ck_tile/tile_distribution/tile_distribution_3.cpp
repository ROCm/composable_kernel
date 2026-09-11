// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding -- C Matrix Register Layout
 *
 * Demonstrates how C-matrix elements are distributed across thread registers
 * after MFMA computation. Unlike A/B (which are DRAM loads), C lives entirely
 * in registers -- the distribution describes which thread holds which output
 * element of C = A x B.
 *
 * This tutorial shows BOTH:
 *   1. The warp-level C distribution (from MFMA m32n32k8 output mapping)
 *   2. The block-level outer distribution (how multiple warps tile C)
 *   3. The composed distribution (what CK actually uses)
 *
 * The macro CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION (default 1) selects
 * between the standard and transposed C register layouts.
 *
 * Tile: M=256 x N=128 (matches the naive GEMM's C block tile)
 * Warp config: MWarp=4, NWarp=1
 * MFMA: m32n32k8 (each warp produces a 32x32 output)
 *
 * No actual MFMA compute -- we construct a C distributed tensor, fill it
 * with marker values (thread_id * 1000 + buffer_index), and print per-thread
 * contents to reveal which buffer slots belong to which thread.
 *
 * Note: Comments and values assume CDNA (warp_size=64). On RDNA (warp_size=32),
 * the thread-to-data mapping will differ.
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include <cstdio>

using namespace ck_tile;

// Controls which C register layout to demonstrate
#ifndef CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
#define CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION 1
#endif

// ============================================================================
// THE GOAL
// ============================================================================
// After GEMM computation, each thread holds a subset of the C matrix
// (M=256 x N=128 = 32768 elements) in its registers. We want to understand
// exactly which C[m][n] elements each thread owns.
//
// The mapping has two levels:
//
// BLOCK LEVEL (256x128 -> warps and iterations):
//   - 4 warps along M (MWarp=4), 1 warp along N (NWarp=1)
//   - Each warp covers 32 M-rows x 128 N-cols of the block tile
//   - Within each warp: MIterPerWarp=2, NIterPerWarp=4
//     -> 2 x 4 = 8 warp-tile iterations per warp
//   - Each warp-tile iteration is a 32x32 MFMA output
//
// WARP LEVEL (32x32 -> threads):
//   - 64 threads produce 32 x 32 = 1024 C elements
//   - Each thread holds 1024/64 = 16 elements
//   - MFMA m32n32k8 arranges these 16 elements in a specific pattern
//
// The per-thread register buffer = 8 iterations x 16 elements = 128 floats.
//
// ============================================================================
// THE SOLUTION: Two-Level Distribution
// ============================================================================
//
// --- WARP-LEVEL C DISTRIBUTION (from MFMA m32n32k8) ---
//
// For fp16->fp32 MFMA m32n32k8 output (kCM0PerLane=4, kCMLane=2,
// kCM1PerLane=4, kCNLane=32):
//
// STANDARD (CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0):
//
//   Hs[0] = sequence<4, 2, 4>   -> M-dim: 4 x 2 x 4 = 32
//   Hs[1] = sequence<32>        -> N-dim: 32
//   Ps_major = tuple<sequence<1, 2>>   -> lane maps to Hs[0][1] and Hs[1][0]
//   Ps_minor = tuple<sequence<1, 0>>
//
//   How to read Ps: the tuple has 1 element -> NDimP=1 -> P0 = lane_id.
//     P0: major=<1,2>, minor=<1,0> -> merged:
//         Hs[0] level 1 -> kCMLane=2  (outer, M-half)
//         Hs[1] level 0 -> kCNLane=32 (inner, N-col -> contiguous!)
//         lane / 32 -> M-half,  lane % 32 -> N-col
//
//   Ys_major = sequence<1, 1>
//   Ys_minor = sequence<0, 2>
//
//   How to read Ys: parallel arrays -- position i gives Yi.
//
//      Ys_major = seq< 1,  1 >   -> Y0 is in Hs[0],  Y1 is in Hs[0]
//      Ys_minor = seq< 0,  2 >   -> Y0 is level 0,   Y1 is level 2
//                     -Y0-  -Y1-
//
//      Y0: Hs[0] level 0 -> kCM0PerLane=4  (M outer per lane)
//      Y1: Hs[0] level 2 -> kCM1PerLane=4  (M inner per lane)
//
//              Hs[0]                   Hs[1]
//         +-----+-----+                 |
//       [Y0]  [P0]   [Y1]            [P0]
//        = 4   = 2    = 4             = 32
//    (M outer)(lane) (M inner)      (lane -> N)
//
//   Per-thread: Y0 x Y1 = 4 x 4 = 16 elements per warp-tile.
//   Lane decomposition: lane / 32 -> M-half (0..1), lane % 32 -> N-col (0..31)
//
// TRANSPOSED (CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=1):
//
//   Hs[0] = sequence<32>          -> First dim: N (swapped!)
//   Hs[1] = sequence<4, 2, 4>    -> Second dim: M (swapped!)
//   Ps_major = tuple<sequence<2, 1>>   -> lane maps to Hs[1][1] and Hs[0][0]
//   Ps_minor = tuple<sequence<1, 0>>
//
//   How to read Ps: tuple has 1 element -> NDimP=1 -> P0 = lane_id.
//     P0: major=<2,1>, minor=<1,0> -> merged:
//         Hs[1] level 1 -> kCMLane=2  (outer, M-half)
//         Hs[0] level 0 -> kCNLane=32 (inner, N-col -> contiguous!)
//         Same lane decomposition as standard, but dimensions are swapped.
//
//   Ys_major = sequence<2, 2>
//   Ys_minor = sequence<0, 2>
//
//   How to read Ys:
//      Ys_major = seq< 2,  2 >   -> Y0 is in Hs[1],  Y1 is in Hs[1]
//      Ys_minor = seq< 0,  2 >   -> Y0 is level 0,   Y1 is level 2
//                     -Y0-  -Y1-
//
//      Y0: Hs[1] level 0 -> kCM0PerLane=4  (M outer per lane)
//      Y1: Hs[1] level 2 -> kCM1PerLane=4  (M inner per lane)
//      Same 16 elements, but now both Y dims are in Hs[1] (M is second).
//
//              Hs[0]               Hs[1]
//                |            +-----+-----+
//              [P0]         [Y0]  [P0]   [Y1]
//              = 32          = 4   = 2    = 4
//           (lane -> N)   (M outer)(lane)(M inner)
//
//   Same 16 elements per thread, but N is the first dimension in the
//   distribution -- this changes which elements are contiguous in the
//   thread buffer, affecting downstream store coalescing.
//
// --- BLOCK-LEVEL OUTER DISTRIBUTION ---
//
//   MIterPerWarp = MPerBlock / (MWarp x WarpGemm::kM) = 256 / (4 x 32) = 2
//   NIterPerWarp = NPerBlock / (NWarp x WarpGemm::kN) = 128 / (1 x 32) = 4
//
//   Hs[0] = sequence<2, 4>  -> M-dim: 2 iters x 4 warps
//   Hs[1] = sequence<4, 1>  -> N-dim: 4 iters x 1 warp
//   Ps_major = tuple<sequence<1, 2>>
//   Ps_minor = tuple<sequence<1, 1>>
//
//   How to read Ps: tuple has 1 element -> NDimP=1 -> P0 = warp_id.
//     P0: major=<1,2>, minor=<1,1> -> merged:
//         Hs[0] level 1 -> MWarp=4 (outer)
//         Hs[1] level 1 -> NWarp=1 (inner, trivial)
//         Total: 4 x 1 = 4 = number of warps
//
//   Ys_major = sequence<1, 2>
//   Ys_minor = sequence<0, 0>
//
//   How to read Ys:
//      Ys_major = seq< 1,  2 >   -> Y0 is in Hs[0],  Y1 is in Hs[1]
//      Ys_minor = seq< 0,  0 >   -> Y0 is level 0,   Y1 is level 0
//                     -Y0-  -Y1-
//
//      Y0: Hs[0] level 0 -> MIterPerWarp=2
//      Y1: Hs[1] level 0 -> NIterPerWarp=4
//      Block-level buffer = Y0 x Y1 = 2 x 4 = 8 warp-tile slots.
//
//   tile_distribution_encoding<sequence<>,
//       tuple<sequence<2, 4>, sequence<4, 1>>,
//       tuple<sequence<1, 2>>, tuple<sequence<1, 1>>,
//       sequence<1, 2>, sequence<0, 0>>
//
// --- COMPOSED (what CK uses) ---
//
//   make_embed_tile_distribution_encoding(block_outer, warp_encoding)
//   embeds the warp encoding inside each (MIter, MWarp, NIter, NWarp) cell.
//   Total per-thread buffer = 2 x 4 x 16 = 128 elements.
//
// ============================================================================

static constexpr index_t kM = 256;
static constexpr index_t kN = 128;

#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
using WarpGemm = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution;
#else
using WarpGemm = WarpGemmMfmaF16F16F32M32N32K8;
#endif

static constexpr index_t kMWarp = 4;
static constexpr index_t kNWarp = 1;

static constexpr index_t kMIterPerWarp = kM / (kMWarp * WarpGemm::kM); // 2
static constexpr index_t kNIterPerWarp = kN / (kNWarp * WarpGemm::kN); // 4

struct TileDistKernelC
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()() const
    {
        static_assert(get_warp_size() == 64,
                      "This tutorial is hard-coded for CDNA (warp_size=64). "
                      "On RDNA (warp_size=32), the encoding values and print logic must change.");

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<kMIterPerWarp, kMWarp>, sequence<kNIterPerWarp, kNWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<float>(c_block_dstr);

        constexpr index_t kBufSize = c_block_tensor.get_thread_buffer_size();

        // Fill each thread's buffer with a marker value:
        // We can't easily set C[m][n] = m*N + n without knowing the inverse mapping,
        // so instead we fill with thread_id * 1000 + buffer_index to identify ownership.
        static_for<0, kBufSize, 1>{}([&](auto i) {
            c_block_tensor.get_thread_buffer()(i) =
                static_cast<float>(threadIdx.x * 1000 + static_cast<int>(i));
        });

        constexpr index_t warp_size = get_warp_size();

        // Copy compile-time-indexed buffer into a plain array for runtime printing
        float local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}(
            [&](auto i) { local_buf[i] = c_block_tensor.get_thread_buffer()[i]; });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                int lane = tid % static_cast<int>(warp_size);
                int warp = tid / static_cast<int>(warp_size);

                printf("Thread %3d  (warp %d, lane %2d)  buf_size=%d\n",
                       tid,
                       warp,
                       lane,
                       static_cast<int>(kBufSize));

#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
                printf("  Layout: TRANSPOSED (N is first dimension)\n");
#else
                printf("  Layout: STANDARD (M is first dimension)\n");
#endif

                printf("  Block-level: MIterPerWarp=%d, NIterPerWarp=%d\n",
                       static_cast<int>(kMIterPerWarp),
                       static_cast<int>(kNIterPerWarp));
                printf("  Warp-level: 16 elements per warp-tile (32x32 MFMA output)\n");
                printf("  Total: %d x %d x 16 = %d elements\n",
                       static_cast<int>(kMIterPerWarp),
                       static_cast<int>(kNIterPerWarp),
                       static_cast<int>(kBufSize));

                constexpr int kPerWarpTile = 16;
                for(int mIter = 0; mIter < static_cast<int>(kMIterPerWarp); mIter++)
                {
                    for(int nIter = 0; nIter < static_cast<int>(kNIterPerWarp); nIter++)
                    {
                        int base = (mIter * static_cast<int>(kNIterPerWarp) + nIter) * kPerWarpTile;
                        printf("  [mIter=%d, nIter=%d] buf[%3d..%3d]:",
                               mIter,
                               nIter,
                               base,
                               base + kPerWarpTile - 1);
                        for(int k = 0; k < kPerWarpTile; k++)
                        {
                            printf(" %.0f", static_cast<double>(local_buf[base + k]));
                        }
                        printf("\n");
                    }
                }
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Tile Distribution: C-Matrix Register Layout ===\n");
                printf("Tile: %dx%d  BlockSize: %d  WarpSize: %d\n",
                       static_cast<int>(kM),
                       static_cast<int>(kN),
                       static_cast<int>(kBlockSize),
                       static_cast<int>(warp_size));
                printf("MWarp=%d, NWarp=%d, MFMA=m32n32k8\n",
                       static_cast<int>(kMWarp),
                       static_cast<int>(kNWarp));
                printf("MIterPerWarp=%d, NIterPerWarp=%d\n",
                       static_cast<int>(kMIterPerWarp),
                       static_cast<int>(kNIterPerWarp));
#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
                printf("Mode: TRANSPOSED C (CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=1)\n");
                printf("  WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution\n");
                printf("  Warp encoding: <seq<>, tuple<seq<32>, seq<4,2,4>>,\n");
                printf("                  tuple<seq<2,1>>, tuple<seq<1,0>>,\n");
                printf("                  seq<2,2>, seq<0,2>>\n");
#else
                printf("Mode: STANDARD C (CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0)\n");
                printf("  WarpGemmMfmaF16F16F32M32N32K8\n");
                printf("  Warp encoding: <seq<>, tuple<seq<4,2,4>, seq<32>>,\n");
                printf("                  tuple<seq<1,2>>, tuple<seq<1,0>>,\n");
                printf("                  seq<1,1>, seq<0,2>>\n");
#endif
                printf("\nBlock outer: <seq<>, tuple<seq<%d,%d>, seq<%d,%d>>,\n",
                       static_cast<int>(kMIterPerWarp),
                       static_cast<int>(kMWarp),
                       static_cast<int>(kNIterPerWarp),
                       static_cast<int>(kNWarp));
                printf("              tuple<seq<1,2>>, tuple<seq<1,1>>,\n");
                printf("              seq<1,2>, seq<0,0>>\n\n");
            }
            __syncthreads();

            // Warp 0, Lane 0
            print_thread(0);
            __syncthreads();
            // Warp 0, Lane 32 (different M-half in standard, different N in transposed)
            print_thread(32);
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 1 (covers different M-rows than warp 0) ---\n");
            __syncthreads();
            print_thread(static_cast<int>(warp_size));
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 3 (last) ---\n");
            __syncthreads();
            print_thread(kBlockSize - 1);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial 3: C-Matrix Register Layout ===\n");
    printf("=== Matches naive GEMM: MPerBlock=256, NPerBlock=128          ===\n\n");
    printf("MFMA m32n32k8: each warp produces 32x32 = 1024 elements\n");
    printf("  64 threads per warp -> 16 elements per thread per warp-tile\n");
    printf("  MWarp=4, NWarp=1 -> 4 warps along M, 1 along N\n");
    printf("  MIterPerWarp=2, NIterPerWarp=4 -> 8 warp-tiles per warp\n");
    printf("  Total per thread: 8 x 16 = 128 elements\n\n");

#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
    printf("Current mode: TRANSPOSED C distribution\n");
    printf("  Rebuild with -DCK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0 for standard\n\n");
#else
    printf("Current mode: STANDARD C distribution\n");
    printf("  Rebuild with -DCK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=1 for transposed\n\n");
#endif

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernelC{}, dim3(1), dim3(TileDistKernelC::kBlockSize), 0));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
