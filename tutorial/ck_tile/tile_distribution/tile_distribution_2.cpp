// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding -- B Matrix DRAM Load
 *
 * Demonstrates how tile_distribution_encoding maps threads to B-matrix
 * elements during a DRAM load in the naive GEMM tutorial.
 *
 * Source: block_gemm_pipeline_agmem_bgmem_creg_policy.hpp
 *         MakeBDramTileDistribution(), with fp16, BlockSize=256
 *
 * Tile: N=128 x K=32  (matches the naive GEMM's B block tile)
 * Threads: 256 (4 warps on CDNA, 8 on RDNA)
 *
 * The B encoding has the SAME structure as the A encoding (Tutorial 1),
 * but with N=128 instead of M=256. This changes only N0 (the iteration
 * count), showing how the same encoding pattern adapts to different
 * tile sizes.
 *
 * No compute is performed -- this is purely about data movement.
 *
 * Note: int32_t is used instead of fp16 for readable printf output.
 * The distribution encoding is hardcoded to match the fp16 derivation.
 *
 * Note: Comments and values assume CDNA (warp_size=64). On RDNA (warp_size=32),
 * the thread-to-data mapping will differ.
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include <cstdio>

using namespace ck_tile;

// ============================================================================
// THE GOAL
// ============================================================================
// Matrix B: N=128 rows x K=32 columns, stored in DRAM (row-major, fp16).
// (In GEMM, B is stored as [N, K] -- each "row" is one output channel.)
// Load the entire tile into registers using 256 threads (4 warps on CDNA).
//
// Same coalescing strategy as the A-matrix (Tutorial 1):
//   - 4 lanes cover one K-row (4 x 8 = 32 K-values)
//   - Each warp (64 lanes) covers 16 N-rows
//   - 4 warps cover 64 N-rows per iteration
//   - N0 = 128/64 = 2 iterations (vs 4 for A's M=256)
//
// Per-thread buffer = 2 iterations x 8 K-values = 16 elements.
//
// Compare with Tutorial 1 (A-matrix):
//   A: M=256, M0=4, buffer=32   |   B: N=128, N0=2, buffer=16
//   Everything else is identical -- same K-splitting, same coalescing.
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding
// ============================================================================
//
//   Production code derives (fp16, BlockSize=256, NPerBlock=128, KPerBlock=32):
//     K1 = 16/sizeof(fp16) = 8
//     K0 = KPerBlock/K1 = 4
//     N2 = warp_size/K0 = 16
//     N1 = BlockSize/warp_size = 4
//     N0 = NPerBlock/(N2*N1) = 2
//
//   Step 1 -- Hierarchical dimensions (Hs):
//
//      Hs[0] = sequence<2, 4, 16>  -> N = 2 x 4 x 16 = 128
//      Hs[1] = sequence<4, 8>      -> K = 4 x 8 = 32
//
//              Hs[0]                    Hs[1]
//         +-----+-----+              +---+---+
//       [Y0]   [P0]  [P1]          [P1]     [Y1]
//        = 2    = 4   = 16          = 4      = 8
//      (iter) (warp) (row)        (K-chunk) (vec load)
//
//   Step 2 -- Parallel dimensions (Ps): NDimP=2 (P0=warp_id, P1=lane_id).
//
//      Ps_major = tuple<sequence<1>, sequence<1, 2>>
//      Ps_minor = tuple<sequence<1>, sequence<2, 0>>
//
//      How to read Ps: the tuple has 2 elements -> NDimP=2.
//        First element  = P0 = warp_id
//        Second element = P1 = lane_id
//
//      P0: major=<1>, minor=<1> -> Hs[0], level 1 -> N1=4 (which warp)
//      P1: major=<1,2>, minor=<2,0> -> merged:
//          Hs[0] level 2 -> N2=16 (outer, row within warp)
//          Hs[1] level 0 -> K0=4  (inner, K-chunk -> coalesced!)
//          lane / 4 -> row_in_warp,  lane % 4 -> K-chunk
//
//   Step 3 -- Yield dimensions (Ys): what each thread owns.
//
//      Ys_major = sequence<1, 2>
//      Ys_minor = sequence<0, 1>
//
//      How to read Ys: parallel arrays -- position i gives Yi.
//
//      Ys_major = seq< 1,  2 >   -> Y0 is in Hs[0],  Y1 is in Hs[1]
//      Ys_minor = seq< 0,  1 >   -> Y0 is level 0,   Y1 is level 1
//                     -Y0-  -Y1-
//
//      Y0: Hs[0] level 0 -> N0=2  (iterations along N)
//      Y1: Hs[1] level 1 -> K1=8  (vector load width)
//
//   Buffer size = Y0 x Y1 = 2 x 8 = 16 elements per thread.
//
// ============================================================================

static constexpr index_t kN = 128;
static constexpr index_t kK = 32;

struct TileDistKernelB
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        static_assert(get_warp_size() == 64,
                      "This tutorial is hard-coded for CDNA (warp_size=64). "
                      "On RDNA (warp_size=32), the encoding values and print logic must change.");

        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kN, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<2, 4, 16>, sequence<4, 8>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});

        auto window = make_tile_window(
            b_tensor, make_tuple(number<kN>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t kBufSize  = 16; // 2 iterations x 8 K-values

        int32_t local_buf[kBufSize];
        static_for<0, kBufSize, 1>{}([&](auto i) { local_buf[i] = static_cast<int32_t>(buf[i]); });

        auto print_thread = [&](int tid) {
            if(static_cast<int>(threadIdx.x) == tid)
            {
                int lane       = tid % static_cast<int>(warp_size);
                int warp       = tid / static_cast<int>(warp_size);
                int row_in_wrp = lane / 4;
                int k_chunk    = lane % 4;

                printf("Thread %3d  (warp %d, lane %2d)  row_in_warp=%2d  k_chunk=%d\n",
                       tid,
                       warp,
                       lane,
                       row_in_wrp,
                       k_chunk);

                for(int iter = 0; iter < 2; iter++)
                {
                    int row = iter * 64 + warp * 16 + row_in_wrp;
                    int col = k_chunk * 8;
                    printf("  iter %d: B[%3d][%2d..%2d] =", iter, row, col, col + 7);
                    for(int k = 0; k < 8; k++)
                        printf(" %4d", local_buf[iter * 8 + k]);
                    printf("\n");
                }
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Tile Distribution: B-Matrix DRAM Load ===\n");
                printf("Source: MakeBDramTileDistribution (fp16, BlockSize=256)\n");
                printf("Tile: %dx%d  BlockSize: %d  WarpSize: %d  Warps: %d\n",
                       kN,
                       kK,
                       kBlockSize,
                       static_cast<int>(warp_size),
                       kBlockSize / static_cast<int>(warp_size));
                printf("Each thread: 2 iterations x 8 K-values = 16 elements\n");
                printf("Compare with Tutorial 1 (A): same K-split, but N0=2 vs M0=4\n\n");
            }
            __syncthreads();

            // Lane 0: row_in_warp=0, k_chunk=0 -> rows {0, 64}, K=0..7
            print_thread(0);
            __syncthreads();
            // Lane 1: k_chunk=1 -> same rows, K=8..15
            print_thread(1);
            __syncthreads();
            // Lane 4: row_in_warp=1 -> rows {1, 65}, K=0..7
            print_thread(4);
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 1 ---\n");
            __syncthreads();
            // Warp 1, Lane 0: rows {16, 80}, K=0..7
            print_thread(static_cast<int>(warp_size));
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 3 (last) ---\n");
            __syncthreads();
            // Warp 3, Lane 63: rows {63, 127}, K=24..31
            print_thread(kBlockSize - 1);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial 2: B-Matrix DRAM Load ===\n");
    printf("=== Matches naive GEMM: NPerBlock=128, KPerBlock=32     ===\n\n");

    HostTensor<int32_t> h_tensor({kN, kK});
    for(int i = 0; i < kN * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix B[%d x %d], row-major, B[n][k] = n*%d + k\n\n", kN, kK, kK);

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernelB{},
                                 dim3(1),
                                 dim3(TileDistKernelB::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
