// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.10: M-vectorized READ + b128 global STORE
 *
 * Motivation
 * ----------
 * In 08 we profiled the following per-thread instruction mix:
 *
 *     1 x buffer_load_dwordx4      (b128 global load)
 *     1 x ds_write_b128            (b128 LDS write, 0 conflicts)
 *     1 x ds_read_b128             (b128 LDS read,  0 conflicts)
 *     8 x buffer_store_short       (narrow global store -- NOT b128)
 *
 * The narrow store is NOT due to a missing descriptor hint. It is a
 * structural consequence of the tile distribution:
 *   - output memory is [K, M] row-major -> M is the contiguous axis
 *   - each thread's 8 fp16 register vector holds 8 K values at 1 M
 *     (i.e. vector is along the K-axis, which is the STRIDED axis of
 *     the output buffer)
 *   - so the 8 fp16 of one thread are stride M apart in memory,
 *     impossible to pack into a single b128 transaction
 *
 * The warp still coalesces the 8 narrow stores into 8 contiguous 128 B
 * HBM transactions, so at M=131072, K=512 we already saturate HBM
 * bandwidth (~6 TB/s). But the 8 separate issue slots cost front-end
 * cycles, and narrow stores can also limit memory-pipe throughput.
 *
 * Fix attempt: rotate the per-thread vector axis
 * ----------------------------------------------
 * To get buffer_store_dwordx4, each thread must hold 8 consecutive
 * M values at a fixed K (vector along M, the contiguous axis of the
 * output). That means changing the READ distribution's per-thread
 * shape from (M=1, K=8) -- used in 07/08 -- to (M=8, K=1).
 *
 * Keep:
 *   - Global [M, K] LOAD : K-vectorized (M-partitioned warps),
 *                          b128 -- unchanged from 08.
 *   - LDS WRITE          : K-vectorized, matches XOR pack axis,
 *                          b128, 0 bank conflicts -- unchanged.
 *   - LDS XOR descriptor : the same [M, K] XOR with pack on K
 *                          (kKPack=8) as 07/08.
 *
 * Change:
 *   - LDS READ distribution: K-partitioned by warp (same cross-warp
 *     exchange as 07/08), but per-thread shape is now (M=8, K=1).
 *     Each thread reads 8 M values at a fixed K. Those 8 fp16 live
 *     at LDS addresses stride kK * sizeof(T) = 64 B apart (plus
 *     XOR scrambling), so they CANNOT be packed into one ds_read_b128.
 *     The compiler emits ~8 narrow ds_read per thread instead.
 *
 *   - Global [K, M] STORE: same [M, K]-shaped window-reinterpret trick
 *     as 08 (logical (m, k) -> physical offset 1*m + M*k in the [K, M]
 *     buffer). With the new per-thread (M=8, K=1) layout, the 8 M
 *     values at 1 K now lie at 8 CONSECUTIVE fp16 in the [K, M] buffer
 *     -> 1 x buffer_store_dwordx4 per thread.
 *
 * Expected per-thread mix:
 *
 *     1 x buffer_load_dwordx4      (b128 global load, same as 08)
 *     1 x ds_write_b128            (b128 LDS write,   same as 08)
 *     N x ds_read_*                (narrow LDS reads, may conflict)
 *     1 x buffer_store_dwordx4     (b128 global store -- the win)
 *
 * Trade-off
 * ---------
 * We traded 8 narrow HBM stores for N narrow LDS reads. LDS is ~10x
 * faster than HBM per transaction, so in principle this should win --
 * IF the LDS reads don't blow up SQ_LDS_BANK_CONFLICT. The XOR
 * descriptor was designed for the K-vectorized read in 07/08; applying
 * it to an M-vectorized read is NOT the pattern it was tuned for. The
 * scrambling still helps (the naive stride-kK read from a non-XOR LDS
 * would hit only kBanks / gcd(kK, kBanks) distinct banks), but we
 * should expect residual conflicts. This example measures how much.
 *
 * A further follow-up would re-tune the XOR to pack on M (kMPack) and
 * swizzle on K, which would give 0 conflicts on the M-vector read at
 * the cost of conflicts on the K-vector write. The fundamentally
 * asymmetric LDS layout means you can pick ONE vector axis to make
 * conflict-free, not both.
 *
 * Measured on MI355 / gfx950, M=131072, K=512, fp16 (HBM saturating):
 *
 *                               time    BW        SQ_INSTS_LDS   SQ_LDS_BANK_CONFLICT
 *   07 xor_cross_warp_lds       44.25us 6067 GB/s         128*             0
 *   08 xor_cross_warp_window    44.25us 6067 GB/s         128*             0
 *   10 m_vector_store (this)    45.14us 5947 GB/s         576*             0
 *                                                         ^ ~4.5x more LDS insts
 *   (* counters taken at the small 256x256 test size: 128 / 256 / 1152
 *    scale linearly with M*K; ratios shown here.)
 *
 * Verdict: the b128 global store does fire (confirmed via
 * `llvm-objdump -d ... | grep buffer_store_dwordx4`), but at
 * HBM-saturating workloads the fused b128 store is NOT a win. The
 * 8x buffer_store_short warp-coalesces to the same HBM bandwidth
 * (6 TB/s is already at the MI355 peak) and the extra ~4x LDS
 * read instructions in this design become a net regression of
 * about 2% vs 07/08. SURPRISINGLY, we still see 0 bank conflicts
 * even though the K-pack XOR wasn't designed for M-vector reads --
 * the swizzle is generous enough to scatter stride-64B addresses
 * across all banks.
 *
 * Takeaway: "make every memory op b128" is the wrong optimization
 * target. What matters is the width of the bottleneck stage. When
 * HBM is saturated and the narrow stores coalesce at the warp
 * level, there is no headroom left on the store side, and shifting
 * work to LDS is a pure loss.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct MVectorStoreKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM         = 64;
    static constexpr index_t kK         = 32;
    static constexpr index_t kKPack     = 8; // LDS XOR pack axis (same as 07/08)
    static constexpr index_t kMPack     = 8; // per-thread register vector on the READ side

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // --------------------------------------------------------------------
    // SAME [M, K] XOR LDS descriptor as 07/08 -- pack on K (kKPack=8),
    // swizzle on M. Tuned for K-vectorized access on both sides. We
    // keep it unchanged here to isolate the effect of rotating the
    // READ per-thread vector axis from K to M.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr index_t NBanks    = get_n_lds_banks(); // 32 on gfx942, 64 on gfx950
        constexpr index_t LdsBandwidthBytes = NBanks * get_n_dwords_per_128b();
        constexpr index_t MLdsLayerRaw      = LdsBandwidthBytes / kK / DataTypeSize;
        constexpr index_t MLdsLayer         = MLdsLayerRaw < 1 ? 1 : MLdsLayerRaw;
        constexpr index_t RowMul            = (NBanks == 64) ? 2 : 1;

        // Step 1: (B, A, C) natural row-major order.
        //   B = M / MLdsLayer       (outermost, stride kK*MLdsLayer)
        //   A = K / KPack * Layer   (middle, stride KPack)
        //   C = KPack               (innermost, stride 1 = vector axis)
        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kM / MLdsLayer>{},           // B
                       number<kK / kKPack * MLdsLayer>{},  // A
                       number<kKPack>{}),                  // C
            make_tuple(number<kK * MLdsLayer>{},           // stride B
                       number<kKPack>{},                   // stride A
                       number<1>{}),                       // stride C
            number<kKPack>{},
            number<1>{});

        // Step 2: XOR on (B, A). xor_t: A_low = A XOR (B % A).
        constexpr auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_tuple(make_xor_transform(
                           make_tuple(number<kM / MLdsLayer * RowMul>{},   // B_eff
                                      number<kK / kKPack * MLdsLayer>{})), // A
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0, 1>{}, sequence<2>{}));

        // Step 3: Unmerge A -> (Layer, K/Pack). Upper: (B=0, L=1, K0=2, C=3).
        constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
            lds_desc_permuted,
            make_tuple(
                make_pass_through_transform(number<kM / MLdsLayer>{}),
                make_unmerge_transform(make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{},    sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Step 4: Merge to [M, K].
        constexpr auto lds_desc = transform_tensor_descriptor(
            lds_desc_unmerged,
            make_tuple(
                make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kK / kKPack>{},    number<kKPack>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{},    sequence<1>{}));

        return lds_desc;
    }

    // --------------------------------------------------------------------
    // WRITE distribution (M-partitioned by warp, K-vectorized per thread)
    // Identical to 07/08 -- we do NOT change the write side.
    //   Warp w covers M=[w*16, w*16+16), all K=[0, 32).
    //   Per-thread Y shape: (M0=1, K1=8)  -- 8 K values at a fixed M.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeWriteDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType); // 8 for fp16
        constexpr index_t K0 = kK / K1;               // 4
        constexpr index_t M2 = 64 / K0;               // 16
        constexpr index_t M1 = kBlockSize / 64;       // 4 (warps)
        constexpr index_t M0 = kM / (M2 * M1);        // 1

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // --------------------------------------------------------------------
    // OUTPUT distribution for the [K, M] global store.
    //
    // Relabels the per-thread (M=8, K=1) layout from the READ distribution
    // onto a [kK, kM]-shaped tile with per-thread (K=1, M=8). Same 8 fp16
    // per thread -- the transpose_tile2d that converts between them is a
    // compile-time relabel (like in 07), NOT a register shuffle.
    //
    // Why we need a SEPARATE [kK, kM] tile: `make_naive_tensor_descriptor`
    // only carries a vectorization guarantee on the LAST dimension. Our
    // output buffer is [K, M] row-major, so the stride-1 axis is M. To
    // get the hint to actually fire (and turn 8x buffer_store_short into
    // 1x buffer_store_dwordx4) we have to put M as the LAST dim of the
    // descriptor and therefore of the tile.
    //
    //   Hs[0] (K-group) = (K_warp=4, K_lane=8, K_per_thread=1)
    //   Hs[1] (M-group) = (M_lane=8, M_per_thread=8)
    //
    //   P0 (warp)          -> Hs[0].minor[0] = K_warp
    //   P1 (lane)          -> (Hs[0].minor[1]=K_lane, Hs[1].minor[0]=M_lane)
    //   Y0 (slow)          -> Hs[0].minor[2] = K_per_thread = 1
    //   Y1 (fast, vector)  -> Hs[1].minor[1] = M_per_thread = 8
    //
    // The (warp -> K_warp, lane -> (K_lane, M_lane)) mapping is IDENTICAL
    // to the READ distribution, so every thread's 8 fp16 land in the same
    // (k, m_start..m_start+7) global position that the read pulled them
    // from. That is why transpose_tile2d can be a no-op between the two.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistributionKM()
    {
        constexpr index_t M_PerThread = kMPack;           // 8
        constexpr index_t M_Lane      = kM / M_PerThread; // 8
        constexpr index_t K_Warp      = kBlockSize / 64;  // 4
        constexpr index_t K_Lane      = 64 / M_Lane;      // 8
        constexpr index_t K_PerThread = kK / (K_Warp * K_Lane); // 1

        // Hs[0]=K-group {K_Warp, K_Lane, K_PerThread}
        // Hs[1]=M-group {M_Lane, M_PerThread}
        //
        // P1 ordering is critical: the READ distribution decomposes a
        // 64-wide lane id as (M_Lane, K_Lane) -- majs (1, 2), mins (0, 1).
        // We must keep that same decomposition here so lane L in both
        // distributions refers to the same (m_lane, k_lane) pair. Then
        // transpose_tile2d between the two is a pure relabel and no
        // cross-lane shuffle is emitted.
        //
        // P1 majs (2, 1) = (M-grp, K-grp); mins (0, 1) = (M_Lane, K_Lane).
        //
        // Y swaps axes vs the read dist: Y0 -> K_PerThread (slow, =1),
        // Y1 -> M_PerThread (fast, vector=8).
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<K_Warp, K_Lane, K_PerThread>,
                                             sequence<M_Lane, M_PerThread>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
    }

    // --------------------------------------------------------------------
    // READ distribution (K-partitioned by warp, M-vectorized per thread).
    //
    // This is the key difference vs 07/08. Cross-warp exchange is the
    // same (warp w reads K=[w*8, w*8+8), all M), but the per-thread
    // register layout is ROTATED 90 degrees:
    //
    //    07/08:  per-thread shape (M=1, K=8)  -- 8 K values at 1 M
    //    10    :  per-thread shape (M=8, K=1)  -- 8 M values at 1 K
    //
    // Tile shape fed to make_tile_window is still [kM, kK] so the same
    // LDS descriptor is reused. We use:
    //
    //   Hs[0] (M-group) = (M_lane = 8, M_per_thread = 8)        -> kM = 64
    //   Hs[1] (K-group) = (K_warp = 4, K_lane = 8, K_per_thread = 1) -> kK = 32
    //
    //   P0 (warp id)            -> Hs[1].minor[0] = K_warp = 4
    //   P1 (lane id, 64 wide)   -> (Hs[0].minor[0] = M_lane = 8,
    //                               Hs[1].minor[1] = K_lane = 8)
    //   Y0 (fast per-thread dim)-> Hs[0].minor[1] = M_per_thread = 8
    //   Y1 (slow per-thread dim)-> Hs[1].minor[2] = K_per_thread = 1
    //
    // Consequence for LDS: the 8 fp16 each thread reads are at
    // LDS[m_start .. m_start+7, k_fixed] which in the naive [M, K]
    // layout live stride kK*sizeof(T) = 64 B apart. They cannot pack
    // into a single ds_read_b128. The compiler emits 8 ds_read_u16
    // (or equivalent) per thread. XOR still scatters those 64 addresses
    // across banks but the scattering was designed for the *other*
    // vector axis, so expect SQ_LDS_BANK_CONFLICT > 0.
    //
    // Consequence for global store: each thread now has its 8 fp16 at
    // 8 consecutive M values for 1 K. In the [K, M] output buffer
    // these are 8 contiguous bytes * 2 = 16 contiguous bytes -> fits
    // in a single buffer_store_dwordx4.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeReadDistributionMK()
    {
        constexpr index_t M_PerThread = kMPack;           // 8
        constexpr index_t M_Lane      = kM / M_PerThread; // 8
        constexpr index_t K_Warp      = kBlockSize / 64;  // 4
        constexpr index_t K_Lane      = 64 / M_Lane;      // 8
        constexpr index_t K_PerThread = kK / (K_Warp * K_Lane); // 1

        static_assert(M_Lane * M_PerThread == kM, "M partition must cover all kM rows");
        static_assert(K_Warp * K_Lane * K_PerThread == kK, "K partition must cover all kK cols");
        static_assert(M_Lane * K_Lane == 64, "lane partition must cover 64 lanes");
        static_assert(K_Warp == kBlockSize / 64, "K_Warp must equal #warps");

        // Hs = (M-group {M_Lane=8, M_PerThread=8},
        //       K-group {K_Warp=4, K_Lane=8, K_PerThread=1})
        //
        // P0 (warp) -> Hs[1].minor[0] = K_Warp            (warp id -> K range)
        // P1 (lane) -> (Hs[0].minor[0]=M_Lane,            (lane -> (M_lane, K_lane))
        //               Hs[1].minor[1]=K_Lane)
        // Y0        -> Hs[0].minor[1] = M_PerThread       (per-thread vec on M)
        // Y1        -> Hs[1].minor[2] = K_PerThread       (per-thread scalar on K)
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M_Lane, M_PerThread>,
                                             sequence<K_Warp, K_Lane, K_PerThread>>,
                                       tuple<sequence<2>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<1, 2>>{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                   DataType* __restrict__ output,
                                   index_t M,
                                   index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M)
            return;

        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);

        constexpr auto dist_mk_write = MakeWriteDistributionMK();
        constexpr auto dist_mk_read  = MakeReadDistributionMK();
        constexpr auto dist_km_out   = MakeOutputDistributionKM();

        auto lds_window_write =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk_write);
        auto lds_window_read =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk_read);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // [1] Global [M, K] load, K-vectorized. Identical to 08.
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<kKPack>{},
                number<1>{});
            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);
            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk_write);

            auto reg_write = load_tile(gmem_window_in);

            // [2] LDS write, unchanged from 08: b128, 0 conflicts.
            store_tile(lds_window_write, reg_write);
            block_sync_lds();

            // [3] LDS read with M-vectorized per-thread shape. Each
            //     thread gathers 8 M values at 1 K. These are stride
            //     kK * sizeof(T) apart in the physical LDS, so the
            //     compiler emits 8 narrow ds_read per thread instead
            //     of 1 ds_read_b128. XOR still scatters across banks
            //     but not optimally for this access axis.
            auto reg_read = load_tile(lds_window_read);
            block_sync_lds();

            // [4] Relabel the register tile from [kM, kK] with per-thread
            //     (M=8, K=1) to [kK, kM] with per-thread (K=1, M=8). The
            //     (warp, lane) mapping is identical in both distributions,
            //     so transpose_tile2d is a compile-time no-op -- no
            //     shuffle ISA emitted (same as in 07). We only do it to
            //     tell `store_tile` to use an output descriptor whose
            //     LAST dim is M, so the vectorization hint fires.
            auto reg_km = make_static_distributed_tensor<DataType>(dist_km_out);
            transpose_tile2d(reg_km, reg_read);

            // [5] Global [K, M] store. Natural descriptor with M as the
            //     contiguous LAST dim so the hint (kMPack=8, stride=1)
            //     takes effect. Per-thread 8 M values at 1 K now pack
            //     into a single buffer_store_dwordx4 -- the whole point
            //     of this example.
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}),
                number<kMPack>{},
                number<1>{});
            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);
            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km_out);

            store_tile(gmem_window_out, reg_km);
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.10: M-vectorized READ + b128 global STORE\n";
    std::cout << "========================================\n\n";

    std::cout << "Same [M, K] XOR LDS as 07/08.\n";
    std::cout << "WRITE distribution: M-partitioned, per-thread (M=1, K=8) -- unchanged.\n";
    std::cout << "READ  distribution: K-partitioned, per-thread (M=8, K=1) -- ROTATED.\n";
    std::cout << "Output:  1x buffer_store_dwordx4 per thread (was 8x buffer_store_short).\n";
    std::cout << "LDS read: 8x narrow ds_read per thread (was 1x ds_read_b128).\n\n";

    constexpr index_t M = 65536;
    constexpr index_t K = 256;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);

    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            h_input[m * K + k] = static_cast<DataType>(m * 10 + k);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM         = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size      = (M + kM - 1) / kM;

    std::cout << "Configuration:\n";
    std::cout << "  Input:  [" << M << ", " << K << "] (row-major)\n";
    std::cout << "  Output: [" << K << ", " << M << "] (transposed)\n";
    if(num_iters > 1)
        std::cout << "  Iterations: " << num_iters << " (warmup: " << num_warmup << ")\n";
    std::cout << "\n";

    stream_config stream;
    constexpr index_t lds_size = MVectorStoreKernel<DataType>::GetStaticLdsSize();

    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          MVectorStoreKernel<DataType>{},
                          dim3(grid_size),
                          dim3(block_size),
                          lds_size,
                          static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                          static_cast<DataType*>(d_output.GetDeviceBuffer()),
                          M,
                          K));
    }
    hip_check_error(hipDeviceSynchronize());

    hipEvent_t start, stop;
    hip_check_error(hipEventCreate(&start));
    hip_check_error(hipEventCreate(&stop));
    hip_check_error(hipEventRecord(start, nullptr));

    for(int i = 0; i < num_iters; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          MVectorStoreKernel<DataType>{},
                          dim3(grid_size),
                          dim3(block_size),
                          lds_size,
                          static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                          static_cast<DataType*>(d_output.GetDeviceBuffer()),
                          M,
                          K));
    }

    hip_check_error(hipEventRecord(stop, nullptr));
    hip_check_error(hipDeviceSynchronize());

    float elapsed_ms = 0;
    hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));
    float avg_time_us = (elapsed_ms * 1000.0f) / num_iters;

    hip_check_error(hipEventDestroy(start));
    hip_check_error(hipEventDestroy(stop));

    d_output.FromDevice(h_output.data(), K * M * sizeof(DataType));

    bool passed         = true;
    index_t error_count = 0;

    for(index_t k = 0; k < K && error_count < 10; ++k)
    {
        for(index_t m = 0; m < M && error_count < 10; ++m)
        {
            DataType expected = h_input[m * K + k];
            DataType actual   = h_output[k * M + m];

            if(bit_cast<uint16_t>(expected) != bit_cast<uint16_t>(actual))
            {
                std::cout << "Error at [" << k << "][" << m << "]: "
                          << "expected " << static_cast<float>(expected)
                          << ", got " << static_cast<float>(actual) << "\n";
                error_count++;
                passed = false;
            }
        }
    }

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";

    if(num_iters > 1)
    {
        std::cout << "\nPerformance:\n";
        std::cout << "  Average time: " << avg_time_us << " us\n";
        float bandwidth_gbs = (M * K * sizeof(DataType) * 2) / (avg_time_us * 1000.0f);
        std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s\n";
    }

    std::cout << "\nCompare against 08 (same workload):\n";
    std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -- "
                 "./aa_tutorial_14_10_m_vector_store --bench\n";
    std::cout << "Then inspect ISA to confirm buffer_store_dwordx4 appears:\n";
    std::cout << "  llvm-objdump -d <hsaco> | grep buffer_store\n";

    return passed;
}

int main(int argc, char* argv[])
{
    int num_iters  = 1;
    int num_warmup = 0;

    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--bench")
        {
            num_iters  = 100;
            num_warmup = 5;
        }
    }

    return run_test(num_iters, num_warmup) ? 0 : 1;
}
