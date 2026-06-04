// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Tutorial: CK Tile Distribution Encoding -- A Matrix DRAM Load
 *
 * Demonstrates how tile_distribution_encoding maps threads to A-matrix
 * elements during a DRAM load in the naive GEMM tutorial.
 *
 * Source: block_gemm_pipeline_agmem_bgmem_creg_policy.hpp
 *         MakeADramTileDistribution(), with fp16, BlockSize=256
 *
 * Tile: M=256 x K=32  (matches the naive GEMM's A block tile)
 * Threads: 256 (4 warps on CDNA, 8 on RDNA)
 *
 * Host initialises A with sequential values 0, 1, 2, ... (row-major).
 * A[m][k] = m * K + k, so the printed value directly gives the linear index.
 * GPU kernel loads A using the distribution, then prints per-thread buffer
 * contents so the reader can verify which elements each thread received.
 *
 * Note: int32_t is used instead of fp16 for readable printf output.
 * The distribution encoding is hardcoded to match the fp16 derivation
 * (K1=16/sizeof(fp16)=8), not recomputed from sizeof(int32_t).
 *
 * No compute is performed -- this is purely about data movement.
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
// Matrix A: M=256 rows x K=32 columns, stored in DRAM (row-major, fp16).
// Load the entire tile into registers using 256 threads (4 warps on CDNA).
//
// For coalesced memory access with fp16, each lane loads 8 contiguous
// K-values (8 x 2 bytes = 16 bytes = 128 bits). Since K=32, we need
// 32/8 = 4 lanes to cover one row:
//
//      lane 0: K=0..7    lane 1: K=8..15    lane 2: K=16..23    lane 3: K=24..31
//      +---------------- one row of 32 K-columns ------------------------------+
//
// With warp_size=64, each warp has 64 lanes. 4 lanes per row means
// 64/4 = 16 rows per warp. With 4 warps, one pass covers 4x16 = 64 rows.
// To cover all 256 rows, each thread iterates M0 = 256/64 = 4 times.
//
// Per-thread buffer = 4 iterations x 8 K-values = 32 elements.
//
// Visually for warp 0 (lanes 0-63):
//
//       A matrix (256x32)           lane_id decomposition
//       ----------------           ----------------------
//       row  0: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L0       L1      L2       L3       <- iter 0
//       row  1: [ K=0..7 | 8..15 | 16..23 | 24..31 ]
//                  L4       L5      L6       L7
//       ...
//       row 15: same pattern, lanes 60-63
//       ------ stride of 64 rows (4 warps x 16 rows/warp) ------
//       row 64: L0..L3                                    <- iter 1
//       ...
//       row 128: L0..L3                                   <- iter 2
//       ...
//       row 192: L0..L3                                   <- iter 3
//
// ============================================================================
// THE SOLUTION: tile_distribution_encoding
// ============================================================================
//
//   Production code derives (fp16, BlockSize=256, MPerBlock=256, KPerBlock=32):
//     K1 = 16/sizeof(fp16) = 8   -> vector load width (8 values)
//     K0 = KPerBlock/K1 = 4      -> 4 K-chunks per row
//     M2 = warp_size/K0 = 16     -> 16 rows per warp
//     M1 = BlockSize/warp_size = 4  -> 4 warps
//     M0 = MPerBlock/(M2*M1) = 4 -> 4 iterations
//
//   Step 1 -- Hierarchical dimensions (Hs): factor each axis.
//
//      Hs[0] = sequence<4, 4, 16>  -> M = 4 x 4 x 16 = 256
//      Hs[1] = sequence<4, 8>      -> K = 4 x 8 = 32
//
//              Hs[0]                    Hs[1]
//         +-----+-----+              +---+---+
//      level 0  level 1  level 2   level 0  level 1
//        = 4     = 4      = 16       = 4      = 8
//
//   Step 2 -- Parallel dimensions (Ps): NDimP=2 (P0=warp_id, P1=lane_id).
//
//      P0 = warp_id  -> Hs[0][1] = 4  (which warp -> which M-group)
//      P1 = lane_id  -> Hs[0][2]=16 AND Hs[1][0]=4  (merged, total=64)
//
//      The merge transform decomposes lane_id:
//        row_in_warp = lane_id / 4   (0..15, outer)
//        k_chunk     = lane_id % 4   (0..3,  inner -> coalesced!)
//
//      Ps_major = tuple<sequence<1>, sequence<1, 2>>
//      Ps_minor = tuple<sequence<1>, sequence<2, 0>>
//
//      How to read Ps: the tuple has 2 elements -> NDimP=2.
//        First element  = P0 = warp_id
//        Second element = P1 = lane_id
//
//      Ps_major = tuple< seq<1>,     seq<1, 2>  >
//                        -P0(warp)-  -P1(lane)--
//      Ps_minor = tuple< seq<1>,     seq<2, 0>  >
//                        -P0(warp)-  -P1(lane)--
//
//      P0: major=<1>, minor=<1> -> Hs[0], level 1 -> M1=4
//      P1: major=<1,2>, minor=<2,0> -> merged:
//          Hs[0] level 2 -> M2=16 (outer, changes slowly)
//          Hs[1] level 0 -> K0=4  (inner, changes every lane -> coalesced!)
//          Total: 16 x 4 = 64 = warp_size
//          lane / 4 -> row_in_warp (M2),  lane % 4 -> K-chunk (K0)
//
//   Step 3 -- Yield dimensions (Ys): what each thread owns.
//
//      Y0 = Hs[0][0] = 4  (M-iterations)
//      Y1 = Hs[1][1] = 8  (vector load width)
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
//      Y0: Hs[0] level 0 -> M0=4  (iterations along M)
//      Y1: Hs[1] level 1 -> K1=8  (vector load width)
//      Buffer size = Y0 x Y1 = 4 x 8 = 32 elements per thread.
//
//   Step 4 -- Replicate: Rs = sequence<1> (trivial, size 1).
//
//   Complete tree:
//
//              Hs[0]                    Hs[1]
//         +-----+-----+              +---+---+
//       [Y0]   [P0]  [P1]          [P1]     [Y1]
//        = 4    = 4   = 16          = 4      = 8
//      (iter) (warp) (row)        (K-chunk) (vec load)
//
//   Buffer size = Y0 x Y1 = 4 x 8 = 32 elements per thread.
//
// ============================================================================

static constexpr index_t kM = 256;
static constexpr index_t kK = 32;

struct TileDistKernelA
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(const int32_t* p_data) const
    {
        static_assert(get_warp_size() == 64,
                      "This tutorial is hard-coded for CDNA (warp_size=64). "
                      "On RDNA (warp_size=32), the encoding values and print logic must change.");

        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_data, make_tuple(kM, kK), make_tuple(kK, 1), number<1>{}, number<1>{});

        constexpr auto distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<4, 4, 16>, sequence<4, 8>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});

        auto window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, distribution);

        const auto tile = load_tile(window);

        const auto& buf             = tile.get_thread_buffer();
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t kBufSize  = 32; // 4 iterations x 8 K-values

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

                for(int iter = 0; iter < 4; iter++)
                {
                    int row = iter * 64 + warp * 16 + row_in_wrp;
                    int col = k_chunk * 8;
                    printf("  iter %d: A[%3d][%2d..%2d] =", iter, row, col, col + 7);
                    for(int k = 0; k < 8; k++)
                        printf(" %5d", local_buf[iter * 8 + k]);
                    printf("\n");
                }
            }
        };

        if(blockIdx.x == 0)
        {
            if(threadIdx.x == 0)
            {
                printf("\n=== Tile Distribution: A-Matrix DRAM Load ===\n");
                printf("Source: MakeADramTileDistribution (fp16, BlockSize=256)\n");
                printf("Tile: %dx%d  BlockSize: %d  WarpSize: %d  Warps: %d\n",
                       kM,
                       kK,
                       kBlockSize,
                       static_cast<int>(warp_size),
                       kBlockSize / static_cast<int>(warp_size));
                printf("Each thread: 4 iterations x 8 K-values = 32 elements\n\n");
                printf("Coalescing: lanes 0-3 read K=0..31 of the same row\n");
                printf("            (4 x 8 = 32 K-values = one full row)\n\n");
            }
            __syncthreads();

            // Lane 0: row_in_warp=0, k_chunk=0 -> rows {0, 64, 128, 192}, K=0..7
            print_thread(0);
            __syncthreads();
            // Lane 1: k_chunk=1 -> same rows, K=8..15 (coalesced with lane 0)
            print_thread(1);
            __syncthreads();
            // Lane 4: row_in_warp=1 -> rows {1, 65, 129, 193}, K=0..7
            print_thread(4);
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 1 ---\n");
            __syncthreads();
            // Warp 1, Lane 0: rows {16, 80, 144, 208}, K=0..7
            print_thread(static_cast<int>(warp_size));
            __syncthreads();

            if(threadIdx.x == 0)
                printf("\n--- Warp 3 (last) ---\n");
            __syncthreads();
            // Warp 3, Lane 63: rows {63, 127, 191, 255}, K=24..31
            print_thread(kBlockSize - 1);
            __syncthreads();
        }
    }
};

int main()
{
    printf("=== CK Tile Distribution Tutorial 1: A-Matrix DRAM Load ===\n");
    printf("=== Matches naive GEMM: MPerBlock=256, KPerBlock=32     ===\n\n");

    HostTensor<int32_t> h_tensor({kM, kK});
    for(int i = 0; i < kM * kK; i++)
        h_tensor.mData[i] = i;

    printf("Host matrix A[%d x %d], row-major, A[m][k] = m*%d + k\n\n", kM, kK, kK);

    DeviceMem d_data(h_tensor);

    launch_kernel(stream_config{},
                  make_kernel<1>(TileDistKernelA{},
                                 dim3(1),
                                 dim3(TileDistKernelA::kBlockSize),
                                 0,
                                 static_cast<const int32_t*>(d_data.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());

    printf("Done.\n");
    return 0;
}
