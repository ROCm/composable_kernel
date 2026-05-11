// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 15: Nine Ways to Call a GEMM
 *
 * Computes D[M,N] = A[M,K] @ B[K,N]  (fp16 inputs, fp32 accumulator/output)
 * using seven approaches across ck_tile abstraction levels:
 *
 *   Approach 1 - BlockGemm (no LDS): lowest level.
 *     Custom device kernel. Global -> register -> BlockGemm. No LDS.
 *
 *   Approach 2 - BlockGemm + LDS: unified attention pattern.
 *     Custom device kernel. B goes global -> LDS -> BlockGemm reads from LDS.
 *     A goes global -> register. Uses BlockGemmARegBSmemCRegV2.
 *
 *   Approach 3 - GemmPipeline from device kernel: batched contraction pattern.
 *     Custom device kernel calling GemmPipelineAgBgCrMem directly.
 *     Pipeline handles LDS staging, K-loop, and BlockGemm internally.
 *
 *   Approach 4 - GemmKernel + V1 pipeline + CShuffleEpilogue (basic invoker).
 *     Host-only. CShuffleEpilogue uses LDS to rearrange data before global store.
 *
 *   Approach 5 - GemmKernel + Mem pipeline + CShuffleEpilogue (universal invoker).
 *     Host-only. Same LDS shuffle epilogue with the Mem pipeline.
 *
 *   Approach 6 - GemmKernel + V1 pipeline + Default2DEpilogue.
 *     Host-only. DefaultGemm2DEpilogue stores directly — no LDS shuffle, zero
 *     epilogue shared memory. Simpler, slightly different store pattern.
 *
 *   Approach 7 - GemmKernel + Mem pipeline + Default2DEpilogue.
 *     Host-only. Same direct-store epilogue with the Mem pipeline.
 *
 *   Approach 8 - BlockGemm + double-buffered LDS.
 *     Custom device kernel. Two LDS buffers for B — ping-pong between them
 *     so the next B tile loads while the current one is used for compute.
 *
 *   Approach 9 - GemmKernel + CompV4 pipeline (DoubleSmemBuffer=true).
 *     Host-only. GemmPipelineAgBgCrCompV4 allocates 2x LDS and truly
 *     double-buffers both A and B, overlapping loads with compute.
 *
 * Layouts: A=RowMajor [M,K], B=ColumnMajor [K,N], D=RowMajor [M,N]
 * Tile: 256x256x64, 2x2 warps (256 threads), MFMA 32x32x16
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

using namespace ck_tile;

// ======================= Shared tile configuration ==========================
using MyGemmShape = TileGemmShape<sequence<256, 256, 64>,
                                  sequence<2, 2, 1>,
                                  sequence<32, 32, 16>>;

static constexpr index_t kBlockSize = MyGemmShape::NumWarps * get_warp_size();
static constexpr index_t kM = MyGemmShape::kM;
static constexpr index_t kN = MyGemmShape::kN;
static constexpr index_t kK = MyGemmShape::kK;

// ============================================================================
//  Approach 1 — BlockGemm, no LDS  (lowest level)
// ============================================================================
struct Approach1_BlockGemm
{
    static constexpr index_t kBlockSize = MyGemmShape::NumWarps * get_warp_size();

    using WarpGemmType = WarpGemmDispatcher<half_t, half_t, float, 32, 32, 16, false>;

    using BGPolicy = BlockGemmARegBRegCRegV2CustomPolicy<
        half_t, half_t, float,
        sequence<2, 2, 1>,
        WarpGemmType,
        GemmLoopOrder::KMN>;

    using BGProblem = BlockGemmProblem<half_t, half_t, float, kBlockSize, MyGemmShape>;
    using BG        = BlockGemmARegBRegCRegV2<BGProblem, BGPolicy>;

    CK_TILE_DEVICE void operator()(const half_t* __restrict__ a_ptr,
                                   const half_t* __restrict__ b_ptr,
                                   float* __restrict__ d_ptr,
                                   index_t M, index_t N, index_t K) const
    {
        const index_t n_tiles = N / kN;
        const index_t i_m     = (get_block_id() / n_tiles) * kM;
        const index_t i_n     = (get_block_id() % n_tiles) * kN;
        if(i_m >= M || i_n >= N)
            return;

        const auto a_view = make_naive_tensor_view<address_space_enum::global>(
            a_ptr, make_tuple(M, K), make_tuple(K, 1), number<16 / sizeof(half_t)>{}, number<1>{});

        const auto bt_view = make_naive_tensor_view<address_space_enum::global>(
            b_ptr, make_tuple(N, K), make_tuple(K, 1), number<16 / sizeof(half_t)>{}, number<1>{});

        constexpr auto a_dist =
            make_static_tile_distribution(BG::MakeABlockDistributionEncode());
        constexpr auto b_dist =
            make_static_tile_distribution(BG::MakeBBlockDistributionEncode());
        constexpr auto c_dist =
            make_static_tile_distribution(BG::MakeCBlockDistributionEncode());

        auto c_tile = BG::MakeCBlockTile();
        set_tile(c_tile, 0.0f);

        constexpr auto block_gemm = BG{};

        for(index_t k_off = 0; k_off < K; k_off += kK)
        {
            const auto a_tile = load_tile(make_tile_window(
                a_view, make_tuple(number<kM>{}, number<kK>{}), {i_m, k_off}, a_dist));

            const auto b_tile = load_tile(make_tile_window(
                bt_view, make_tuple(number<kN>{}, number<kK>{}), {i_n, k_off}, b_dist));

            block_gemm(c_tile, a_tile, b_tile);
        }

        auto d_view = make_naive_tensor_view<address_space_enum::global>(
            d_ptr, make_tuple(M, N), make_tuple(N, 1), number<16 / sizeof(half_t)>{}, number<1>{});

        auto d_win = make_tile_window(
            d_view, make_tuple(number<kM>{}, number<kN>{}), {i_m, i_n}, c_dist);
        store_tile(d_win, c_tile);
    }
};

// ============================================================================
//  Approach 2 — BlockGemm + LDS  (unified attention / FMHA pattern)
// ============================================================================
// B is staged through LDS: global -> register (copy dist) -> LDS -> BlockGemm.
// A is loaded directly to registers.
// Uses BlockGemmARegBSmemCRegV2: A in registers, B read from LDS by BlockGemm.
// ============================================================================
struct Approach2_BlockGemmLds
{
    static constexpr index_t kBlockSize = MyGemmShape::NumWarps * get_warp_size();

    using WarpGemmType = WarpGemmDispatcher<half_t, half_t, float, 32, 32, 16, false>;

    using BGPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        half_t, half_t, float,
        sequence<2, 2, 1>,    // BlockWarps M, N, K
        WarpGemmType>;

    using BGProblem = BlockGemmProblem<half_t, half_t, float, kBlockSize, MyGemmShape>;
    using BG        = BlockGemmARegBSmemCRegV2<BGProblem, BGPolicy>;

    // LDS size for B tile: [kN, kK] packed
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kN * kK * sizeof(half_t);
    }
    // just to force recompile (3)
    // Cooperative copy distribution for B: all threads load, no replication
    CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(half_t);               // 8 (vector size)
        constexpr index_t K0 = kK / K1;                           // 8
        constexpr index_t N2 = get_warp_size() / K0;              // 8
        constexpr index_t N1 = kBlockSize / get_warp_size();      // 4
        constexpr index_t N0 = kN / (N2 * N1);                    // 8

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_DEVICE void operator()(const half_t* __restrict__ a_ptr,
                                   const half_t* __restrict__ b_ptr,
                                   float* __restrict__ d_ptr,
                                   index_t M, index_t N, index_t K) const
    {
        __shared__ char smem[GetSmemSize()];
        auto* p_b_lds = reinterpret_cast<half_t*>(smem);

        const index_t n_tiles = N / kN;
        const index_t i_m     = (get_block_id() / n_tiles) * kM;
        const index_t i_n     = (get_block_id() % n_tiles) * kN;
        if(i_m >= M || i_n >= N)
            return;

        // --- Global views ---
        const auto a_view = make_naive_tensor_view<address_space_enum::global>(
            a_ptr, make_tuple(M, K), make_tuple(K, 1),
            number<16 / sizeof(half_t)>{}, number<1>{});

        const auto b_view = make_naive_tensor_view<address_space_enum::global>(
            b_ptr, make_tuple(N, K), make_tuple(K, 1),
            number<16 / sizeof(half_t)>{}, number<1>{});

        // --- LDS view for B: [kN, kK] packed ---
        const auto b_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kN>{}, number<kK>{}));
        auto b_lds_view = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

        // --- Distributions ---
        constexpr auto a_dist = BG::MakeABlockTileDistribution();
        constexpr auto b_copy_dist = MakeBCopyDistribution();

        // --- Copy windows for B (cooperative global -> LDS) ---
        auto b_copy_global_win = make_tile_window(
            b_view, make_tuple(number<kN>{}, number<kK>{}), {i_n, 0}, b_copy_dist);

        auto b_copy_lds_win = make_tile_window(
            b_lds_view, make_tuple(number<kN>{}, number<kK>{}), {0, 0}, b_copy_dist);

        // --- B GEMM window: plain LDS window (BlockGemm adds its own warp distribution) ---
        auto b_gemm_lds_win = make_tile_window(
            b_lds_view, make_tuple(number<kN>{}, number<kK>{}), {0, 0});

        // --- Accumulator ---
        auto c_tile = BG::MakeCBlockTile();
        set_tile(c_tile, 0.0f);

        constexpr auto block_gemm = BG{};

        // --- K-loop ---
        for(index_t k_off = 0; k_off < K; k_off += kK)
        {
            // Phase 1: Load B from global (copy dist) and store to LDS
            const auto b_copy_tile = load_tile(b_copy_global_win);
            store_tile(b_copy_lds_win, b_copy_tile);

            block_sync_lds();

            // Phase 2: Load A from global directly to registers
            const auto a_tile = load_tile(make_tile_window(
                a_view, make_tuple(number<kM>{}, number<kK>{}), {i_m, k_off}, a_dist));

            // Phase 3: BlockGemm — A from registers, B read from LDS by block_gemm
            block_gemm(c_tile, a_tile, b_gemm_lds_win);

            // Phase 4: Advance B window
            if(k_off + kK < K)
            {
                block_sync_lds();
                move_tile_window(b_copy_global_win, {0, kK});
            }
        }

        // --- Store C to global ---
        constexpr auto c_dist =
            remove_cvref_t<decltype(c_tile)>::get_tile_distribution();

        auto d_view = make_naive_tensor_view<address_space_enum::global>(
            d_ptr, make_tuple(M, N), make_tuple(N, 1), number<1>{}, number<1>{});

        auto d_win = make_tile_window(
            d_view, make_tuple(number<kM>{}, number<kN>{}), {i_m, i_n}, c_dist);
        store_tile(d_win, c_tile);
    }
};

// ============================================================================
//  Approach 3 — GemmPipeline from device kernel  (batched contraction pattern)
// ============================================================================
// Custom device kernel that calls GemmPipelineAgBgCrMem directly.
// Pipeline handles LDS staging, K-loop, and BlockGemm internally.
// You manage: tensor views, tile windows, shared memory, store.
// ============================================================================
struct Approach3_PipelineKernel
{
    static constexpr index_t kBlockSize = MyGemmShape::NumWarps * get_warp_size();

    using Traits = TileGemmUniversalTraits<
        false, false, false,
        false,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor>;

    using PipelineProblem = UniversalGemmPipelineProblem<
        half_t, half_t, float, MyGemmShape, Traits>;

    using Pipeline = GemmPipelineAgBgCrMem<PipelineProblem>;

    CK_TILE_DEVICE void operator()(const half_t* __restrict__ a_ptr,
                                   const half_t* __restrict__ b_ptr,
                                   float* __restrict__ d_ptr,
                                   index_t M, index_t N, index_t K) const
    {
        __shared__ char smem[Pipeline::GetSmemSize()];

        const index_t n_tiles = N / kN;
        const index_t i_m     = (get_block_id() / n_tiles) * kM;
        const index_t i_n     = (get_block_id() % n_tiles) * kN;
        if(i_m >= M || i_n >= N)
            return;

        // Row-major A[M,K]: descriptor (M, K) with strides (K, 1)
        auto a_desc = make_naive_tensor_descriptor(
            make_tuple(M, K), make_tuple(K, 1),
            number<Pipeline::GetVectorSizeA()>{}, number<1>{});
        auto a_view = make_tensor_view<address_space_enum::global>(a_ptr, a_desc);

        // Column-major B as [N,K]: descriptor (N, K) with strides (K, 1)
        auto b_desc = make_naive_tensor_descriptor(
            make_tuple(N, K), make_tuple(K, 1),
            number<Pipeline::GetVectorSizeB()>{}, number<1>{});
        auto b_view = make_tensor_view<address_space_enum::global>(b_ptr, b_desc);

        // Pad views (matches UniversalGemmKernel::MakeABlockWindows / MakeBBlockWindows)
        auto a_pad = pad_tensor_view(
            a_view,
            make_tuple(number<kM>{}, number<kK>{}),
            sequence<false, Pipeline::kPadK>{});
        auto b_pad = pad_tensor_view(
            b_view,
            make_tuple(number<kN>{}, number<kK>{}),
            sequence<false, Pipeline::kPadK>{});

        auto a_win = make_tile_window(
            a_pad, make_tuple(number<kM>{}, number<kK>{}), {i_m, 0});
        auto b_win = make_tile_window(
            b_pad, make_tuple(number<kN>{}, number<kK>{}), {i_n, 0});

        const index_t num_loop = K / kK;

        // Pipeline call: handles LDS staging, K-loop, BlockGemm internally
        auto c_tile = Pipeline{}(a_win, b_win, num_loop, static_cast<void*>(smem));

        // Store C to global
        auto d_desc = make_naive_tensor_descriptor(
            make_tuple(M, N), make_tuple(N, 1), number<1>{}, number<1>{});
        auto d_view = make_tensor_view<address_space_enum::global>(d_ptr, d_desc);

        constexpr auto c_dist =
            remove_cvref_t<decltype(c_tile)>::get_tile_distribution();

        auto d_win = make_tile_window(
            d_view, make_tuple(number<kM>{}, number<kN>{}), {i_m, i_n}, c_dist);
        store_tile(d_win, c_tile);
    }
};

// ============================================================================
//  Approach 4 — GemmKernel + V1 pipeline  (basic invoker, host-only)
// ============================================================================
namespace approach4 {

using Traits = TileGemmTraits<
    false, false, false,
    tensor_layout::gemm::RowMajor,
    tensor_layout::gemm::ColumnMajor,
    tensor_layout::gemm::RowMajor>;

using PipelineProblem =
    GemmPipelineProblem<half_t, half_t, float, MyGemmShape, Traits,
                        element_wise::PassThrough, element_wise::PassThrough,
                        half_t>;

using Pipeline    = GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
using Partitioner = GemmTile1DPartitioner<MyGemmShape>;

using EpilogueProblem = CShuffleEpilogueProblem<
    half_t, half_t, tuple<>, float, float, tuple<>,
    tensor_layout::gemm::RowMajor, element_wise::PassThrough,
    Partitioner::MPerBlock, Partitioner::NPerBlock,
    2, 2, 32, 32, 16, PipelineProblem::TransposeC>;

using Epilogue = CShuffleEpilogue<EpilogueProblem>;
using Kernel   = GemmKernel<Partitioner, Pipeline, Epilogue>;

} // namespace approach4

// ============================================================================
//  Approach 5 — GemmKernel + Mem pipeline  (universal invoker, host-only)
// ============================================================================
namespace approach5 {

using Traits = TileGemmUniversalTraits<
    false, false, false, false,
    tensor_layout::gemm::RowMajor,
    tensor_layout::gemm::ColumnMajor,
    tensor_layout::gemm::RowMajor>;

using PipelineProblem = UniversalGemmPipelineProblem<
    half_t, half_t, float, MyGemmShape, Traits>;

using Pipeline    = GemmPipelineAgBgCrMem<PipelineProblem>;
using Partitioner = GemmTile1DPartitioner<MyGemmShape>;

using EpilogueProblem = CShuffleEpilogueProblem<
    half_t, half_t, tuple<>, float, float, tuple<>,
    tensor_layout::gemm::RowMajor, element_wise::PassThrough,
    Partitioner::MPerBlock, Partitioner::NPerBlock,
    2, 2, 32, 32, 16, PipelineProblem::TransposeC>;

using Epilogue = CShuffleEpilogue<EpilogueProblem>;
using Kernel   = GemmKernel<Partitioner, Pipeline, Epilogue>;

} // namespace approach5

// ============================================================================
//  Approach 6 — GemmKernel + V1 pipeline + DefaultGemm2DEpilogue  (host-only)
// ============================================================================
// Same pipeline as approach 4, but DefaultGemm2DEpilogue stores the C tile
// directly to global memory — no LDS shuffle. GetSmemSize() == 0.
// ============================================================================
namespace approach6 {

using Traits = TileGemmTraits<
    false, false, false,
    tensor_layout::gemm::RowMajor,
    tensor_layout::gemm::ColumnMajor,
    tensor_layout::gemm::RowMajor>;

using PipelineProblem =
    GemmPipelineProblem<half_t, half_t, float, MyGemmShape, Traits,
                        element_wise::PassThrough, element_wise::PassThrough,
                        half_t>;

using Pipeline    = GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
using Partitioner = GemmTile1DPartitioner<MyGemmShape>;

using EpilogueProblem = DefaultGemm2DEpilogueProblem<
    half_t, half_t, tuple<>, float, float, tuple<>,
    tensor_layout::gemm::RowMajor, element_wise::PassThrough,
    Partitioner::MPerBlock, Partitioner::NPerBlock,
    false, false,
    32, 32, 16,
    PipelineProblem::TransposeC>;

using Epilogue = DefaultGemm2DEpilogue<EpilogueProblem>;
using Kernel   = GemmKernel<Partitioner, Pipeline, Epilogue>;

} // namespace approach6

// ============================================================================
//  Approach 7 — GemmKernel + Mem pipeline + DefaultGemm2DEpilogue  (host-only)
// ============================================================================
// Same pipeline as approach 5, but DefaultGemm2DEpilogue for direct store.
// ============================================================================
namespace approach7 {

using Traits = TileGemmUniversalTraits<
    false, false, false, false,
    tensor_layout::gemm::RowMajor,
    tensor_layout::gemm::ColumnMajor,
    tensor_layout::gemm::RowMajor>;

using PipelineProblem = UniversalGemmPipelineProblem<
    half_t, half_t, float, MyGemmShape, Traits>;

using Pipeline    = GemmPipelineAgBgCrMem<PipelineProblem>;
using Partitioner = GemmTile1DPartitioner<MyGemmShape>;

using EpilogueProblem = DefaultGemm2DEpilogueProblem<
    half_t, half_t, tuple<>, float, float, tuple<>,
    tensor_layout::gemm::RowMajor, element_wise::PassThrough,
    Partitioner::MPerBlock, Partitioner::NPerBlock,
    false, false,
    32, 32, 16,
    PipelineProblem::TransposeC>;

using Epilogue = DefaultGemm2DEpilogue<EpilogueProblem>;
using Kernel   = GemmKernel<Partitioner, Pipeline, Epilogue>;

} // namespace approach7

// ============================================================================
//  Approach 8 — BlockGemm + double-buffered LDS  (ping-pong)
// ============================================================================
// Same as approach 2, but with TWO LDS buffers for B. While computing from
// buffer 0, the next B tile is loaded and written to buffer 1, then swap.
// This overlaps global-memory latency with compute.
// ============================================================================
struct Approach8_BlockGemmLdsDouble
{
    static constexpr index_t kBlockSize = MyGemmShape::NumWarps * get_warp_size();

    using WarpGemmType = WarpGemmDispatcher<half_t, half_t, float, 32, 32, 16, false>;

    using BGPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        half_t, half_t, float,
        sequence<2, 2, 1>,
        WarpGemmType>;

    using BGProblem = BlockGemmProblem<half_t, half_t, float, kBlockSize, MyGemmShape>;
    using BG        = BlockGemmARegBSmemCRegV2<BGProblem, BGPolicy>;

    static constexpr index_t kSingleBuf = kN * kK * sizeof(half_t);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * kSingleBuf;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(half_t);
        constexpr index_t K0 = kK / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kN / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_DEVICE void operator()(const half_t* __restrict__ a_ptr,
                                   const half_t* __restrict__ b_ptr,
                                   float* __restrict__ d_ptr,
                                   index_t M, index_t N, index_t K) const
    {
        __shared__ char smem[GetSmemSize()];
        auto* p_b_lds_0 = reinterpret_cast<half_t*>(smem);
        auto* p_b_lds_1 = reinterpret_cast<half_t*>(smem + kSingleBuf);

        const index_t n_tiles = N / kN;
        const index_t i_m     = (get_block_id() / n_tiles) * kM;
        const index_t i_n     = (get_block_id() % n_tiles) * kN;
        if(i_m >= M || i_n >= N)
            return;

        const auto a_view = make_naive_tensor_view<address_space_enum::global>(
            a_ptr, make_tuple(M, K), make_tuple(K, 1), 
            number<16 / sizeof(half_t)>{}, number<1>{});
        const auto b_view = make_naive_tensor_view<address_space_enum::global>(
            b_ptr, make_tuple(N, K), make_tuple(K, 1),
            number<16 / sizeof(half_t)>{}, number<1>{});

        const auto b_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kN>{}, number<kK>{}));
        auto b_lds_view_0 = make_tensor_view<address_space_enum::lds>(p_b_lds_0, b_lds_desc);
        auto b_lds_view_1 = make_tensor_view<address_space_enum::lds>(p_b_lds_1, b_lds_desc);

        constexpr auto a_dist      = BG::MakeABlockTileDistribution();
        constexpr auto b_copy_dist = MakeBCopyDistribution();

        auto b_copy_global_win = make_tile_window(
            b_view, make_tuple(number<kN>{}, number<kK>{}), {i_n, 0}, b_copy_dist);

        auto b_copy_lds_win_0 = make_tile_window(
            b_lds_view_0, make_tuple(number<kN>{}, number<kK>{}), {0, 0}, b_copy_dist);
        auto b_copy_lds_win_1 = make_tile_window(
            b_lds_view_1, make_tuple(number<kN>{}, number<kK>{}), {0, 0}, b_copy_dist);

        auto b_gemm_lds_win_0 = make_tile_window(
            b_lds_view_0, make_tuple(number<kN>{}, number<kK>{}), {0, 0});
        auto b_gemm_lds_win_1 = make_tile_window(
            b_lds_view_1, make_tuple(number<kN>{}, number<kK>{}), {0, 0});

        auto c_tile = BG::MakeCBlockTile();
        set_tile(c_tile, 0.0f);
        constexpr auto block_gemm = BG{};
        const index_t num_k_loops = K / kK;

        // Pre-fill buffer 0
        auto b_prefetch = load_tile(b_copy_global_win);
        store_tile(b_copy_lds_win_0, b_prefetch);
        move_tile_window(b_copy_global_win, {0, kK});

        for(index_t k_iter = 0; k_iter < num_k_loops; k_iter += 2)
        {
            // --- Even: compute from buf 0, prefetch to buf 1 ---
            block_sync_lds();

            if(k_iter + 1 < num_k_loops)
                b_prefetch = load_tile(b_copy_global_win);

            const auto a_tile_0 = load_tile(make_tile_window(
                a_view, make_tuple(number<kM>{}, number<kK>{}),
                {i_m, k_iter * kK}, a_dist));
            block_gemm(c_tile, a_tile_0, b_gemm_lds_win_0);

            if(k_iter + 1 < num_k_loops)
            {
                block_sync_lds();
                store_tile(b_copy_lds_win_1, b_prefetch);
                move_tile_window(b_copy_global_win, {0, kK});
            }

            // --- Odd: compute from buf 1, prefetch to buf 0 ---
            if(k_iter + 1 < num_k_loops)
            {
                block_sync_lds();

                if(k_iter + 2 < num_k_loops)
                    b_prefetch = load_tile(b_copy_global_win);

                const auto a_tile_1 = load_tile(make_tile_window(
                    a_view, make_tuple(number<kM>{}, number<kK>{}),
                    {i_m, (k_iter + 1) * kK}, a_dist));
                block_gemm(c_tile, a_tile_1, b_gemm_lds_win_1);

                if(k_iter + 2 < num_k_loops)
                {
                    block_sync_lds();
                    store_tile(b_copy_lds_win_0, b_prefetch);
                    move_tile_window(b_copy_global_win, {0, kK});
                }
            }
        }

        constexpr auto c_dist =
            remove_cvref_t<decltype(c_tile)>::get_tile_distribution();
        auto d_view = make_naive_tensor_view<address_space_enum::global>(
            d_ptr, make_tuple(M, N), make_tuple(N, 1), number<1>{}, number<1>{});
        auto d_win = make_tile_window(
            d_view, make_tuple(number<kM>{}, number<kN>{}), {i_m, i_n}, c_dist);
        store_tile(d_win, c_tile);
    }
};

// ============================================================================
//  Approach 9 — GemmKernel + CompV4 pipeline (true double LDS buffer)
// ============================================================================
// GemmPipelineAgBgCrCompV4 requires DoubleSmemBuffer=true and allocates 2x LDS.
// While BlockGemm reads from one LDS buffer, the next tile is written to the
// other. This is true hardware double buffering at the pipeline level.
// ============================================================================
namespace approach9 {

using Traits = TileGemmUniversalTraits<
    false, false, false,
    true,                                // DoubleSmemBuffer = true
    tensor_layout::gemm::RowMajor,
    tensor_layout::gemm::ColumnMajor,
    tensor_layout::gemm::RowMajor>;

using PipelineProblem = UniversalGemmPipelineProblem<
    half_t, half_t, float, MyGemmShape, Traits>;

using Pipeline    = GemmPipelineAgBgCrCompV4<PipelineProblem>;
using Partitioner = GemmTile1DPartitioner<MyGemmShape>;

using EpilogueProblem = DefaultGemm2DEpilogueProblem<
    half_t, half_t, tuple<>, float, float, tuple<>,
    tensor_layout::gemm::RowMajor, element_wise::PassThrough,
    Partitioner::MPerBlock, Partitioner::NPerBlock,
    false, false,
    32, 32, 16,
    PipelineProblem::TransposeC>;

using Epilogue = DefaultGemm2DEpilogue<EpilogueProblem>;
using Kernel   = GemmKernel<Partitioner, Pipeline, Epilogue>;

} // namespace approach9

// ============================================================================
//  CPU reference, helpers
// ============================================================================
static void reference_gemm(const std::vector<half_t>& a,
                            const std::vector<half_t>& b_col,
                            std::vector<float>& d,
                            index_t M, index_t N, index_t K)
{
    for(index_t m = 0; m < M; ++m)
        for(index_t n = 0; n < N; ++n)
        {
            float acc = 0;
            for(index_t k = 0; k < K; ++k)
                acc += static_cast<float>(a[m * K + k]) *
                       static_cast<float>(b_col[k + n * K]);
            d[m * N + n] = acc;
        }
}

template <typename T>
static void fill_random(std::vector<T>& v, float lo = -1.f, float hi = 1.f)
{
    for(auto& x : v)
        x = static_cast<T>(lo + (hi - lo) * static_cast<float>(rand()) / RAND_MAX);
}

static bool verify(const std::vector<float>& ref,
                   const std::vector<float>& test,
                   float tol = 5e-2f)
{
    index_t errs = 0;
    float max_err = 0;
    for(size_t i = 0; i < ref.size(); ++i)
    {
        float e = std::abs(ref[i] - test[i]);
        max_err = std::max(max_err, e);
        if(e > tol)
        {
            if(errs < 3)
                std::cout << "  mismatch [" << i << "]: " << test[i]
                          << " vs ref " << ref[i] << "  (err " << e << ")\n";
            ++errs;
        }
    }
    std::cout << "  max error: " << max_err;
    if(errs)
        std::cout << "  (" << errs << " errors)";
    std::cout << "\n";
    return errs == 0;
}

// Launch + time a custom device kernel
template <typename Functor>
static void run_custom_kernel(const char* name,
                              Functor func,
                              index_t grid_size,
                              const half_t* a_dev, const half_t* b_dev,
                              float* d_dev,
                              const std::vector<float>& h_d_ref,
                              std::vector<float>& h_d,
                              double gflop,
                              index_t M, index_t N, index_t K)
{
    std::cout << name << "\n";

    stream_config stream;
    constexpr int kWarmup = 5;
    constexpr int kRepeat = 20;

    auto run = [&]() {
        launch_kernel(stream,
                      make_kernel<kBlockSize>(
                          func, dim3(grid_size), dim3(kBlockSize), 0,
                          a_dev, b_dev, d_dev, M, N, K));
    };

    for(int i = 0; i < kWarmup; ++i) run();
    hip_check_error(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    hip_check_error(hipEventCreate(&ev0));
    hip_check_error(hipEventCreate(&ev1));
    hip_check_error(hipEventRecord(ev0));
    for(int i = 0; i < kRepeat; ++i) run();
    hip_check_error(hipEventRecord(ev1));
    hip_check_error(hipEventSynchronize(ev1));

    float ms = 0;
    hip_check_error(hipEventElapsedTime(&ms, ev0, ev1));
    double avg = ms / kRepeat;
    hip_check_error(hipEventDestroy(ev0));
    hip_check_error(hipEventDestroy(ev1));

    hip_check_error(hipMemcpy(h_d.data(), d_dev,
                              M * N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = verify(h_d_ref, h_d);
    std::cout << "  " << (ok ? "PASSED" : "FAILED")
              << "  |  " << avg << " ms  |  "
              << gflop / avg << " TFLOPS\n\n";
}

// Launch + time a GemmKernel (host-only approach)
template <typename Kernel>
static void run_gemm_kernel(const char* name,
                            const GemmHostArgs& host_args,
                            float* d_dev,
                            const std::vector<float>& h_d_ref,
                            std::vector<float>& h_d,
                            double gflop,
                            index_t M, index_t N)
{
    std::cout << name << "\n";

    auto kargs = Kernel::MakeKernelArgs(host_args);
    const dim3 grid  = Kernel::GridSize(host_args.M, host_args.N, host_args.k_batch);
    const dim3 block = Kernel::BlockSize();
    std::cout << "  grid: " << grid.x << "  block: " << block.x << "\n";

    stream_config stream;
    constexpr int kWarmup = 5;
    constexpr int kRepeat = 20;

    auto run = [&]() {
        launch_kernel(stream, make_kernel<0>(Kernel{}, grid, block, 0, kargs));
    };

    for(int i = 0; i < kWarmup; ++i) run();
    hip_check_error(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    hip_check_error(hipEventCreate(&ev0));
    hip_check_error(hipEventCreate(&ev1));
    hip_check_error(hipEventRecord(ev0));
    for(int i = 0; i < kRepeat; ++i) run();
    hip_check_error(hipEventRecord(ev1));
    hip_check_error(hipEventSynchronize(ev1));

    float ms = 0;
    hip_check_error(hipEventElapsedTime(&ms, ev0, ev1));
    double avg = ms / kRepeat;
    hip_check_error(hipEventDestroy(ev0));
    hip_check_error(hipEventDestroy(ev1));

    hip_check_error(hipMemcpy(h_d.data(), d_dev,
                              M * N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = verify(h_d_ref, h_d);
    std::cout << "  " << (ok ? "PASSED" : "FAILED")
              << "  |  " << avg << " ms  |  "
              << gflop / avg << " TFLOPS\n\n";
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char* argv[])
{
    index_t M = 1024, N = 1024, K = 1024;
    if(argc >= 4)
    {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    std::cout << "\n============================================\n"
              << "Tutorial 15: Nine Ways to Call a GEMM\n"
              << "============================================\n"
              << "Problem: " << M << " x " << N << " x " << K << "\n"
              << "Tile:    " << kM << " x " << kN << " x " << kK
              << "  (" << MyGemmShape::NumWarps << " warps, "
              << kBlockSize << " threads)\n"
              << "Layout:  A=RowMajor  B=ColumnMajor  D=RowMajor\n\n";

    const index_t stride_A = K;
    const index_t stride_B = K;
    const index_t stride_D = N;
    const double gflop = 2.0 * M * N * K / 1e9;
    const index_t grid_size = (M / kM) * (N / kN);

    std::vector<half_t> h_a(M * K), h_b(K * N);
    std::vector<float>  h_d_ref(M * N);
    std::vector<float>  h_d(M * N);

    srand(42);
    fill_random(h_a, -0.5f, 0.5f);
    fill_random(h_b, -0.5f, 0.5f);

    std::cout << "Running CPU reference ...";
    std::cout.flush();
    reference_gemm(h_a, h_b, h_d_ref, M, N, K);
    std::cout << " done.\n\n";

    DeviceMem d_a(M * K * sizeof(half_t));
    DeviceMem d_b(K * N * sizeof(half_t));
    DeviceMem d_d(M * N * sizeof(float));

    d_a.ToDevice(h_a.data(), M * K * sizeof(half_t));
    d_b.ToDevice(h_b.data(), K * N * sizeof(half_t));

    const auto* a_dev = static_cast<const half_t*>(d_a.GetDeviceBuffer());
    const auto* b_dev = static_cast<const half_t*>(d_b.GetDeviceBuffer());
    auto* d_dev       = static_cast<float*>(d_d.GetDeviceBuffer());

    // Approach 1 — BlockGemm, no LDS
    run_custom_kernel("Approach 1: BlockGemm (no LDS)",
                      Approach1_BlockGemm{}, grid_size,
                      a_dev, b_dev, d_dev, h_d_ref, h_d, gflop, M, N, K);

    // Approach 2 — BlockGemm + LDS (unified attention pattern)
    run_custom_kernel("Approach 2: BlockGemm + LDS (B staged through LDS)",
                      Approach2_BlockGemmLds{}, grid_size,
                      a_dev, b_dev, d_dev, h_d_ref, h_d, gflop, M, N, K);

    // Approach 3 — GemmPipeline from device kernel
    run_custom_kernel("Approach 3: GemmPipelineAgBgCrMem (device kernel)",
                      Approach3_PipelineKernel{}, grid_size,
                      a_dev, b_dev, d_dev, h_d_ref, h_d, gflop, M, N, K);

    // Approach 4 — GemmKernel + V1 pipeline + CShuffleEpilogue (host-only)
    {
        GemmHostArgs host_args{a_dev, b_dev, d_dev,
                               1, M, N, K, stride_A, stride_B, stride_D};
        run_gemm_kernel<approach4::Kernel>(
            "Approach 4: GemmKernel + V1 + CShuffleEpilogue",
            host_args, d_dev, h_d_ref, h_d, gflop, M, N);
    }

    // Approach 5 — GemmKernel + Mem pipeline + CShuffleEpilogue (host-only)
    {
        GemmHostArgs host_args{a_dev, b_dev, d_dev,
                               1, M, N, K, stride_A, stride_B, stride_D};
        run_gemm_kernel<approach5::Kernel>(
            "Approach 5: GemmKernel + Mem + CShuffleEpilogue",
            host_args, d_dev, h_d_ref, h_d, gflop, M, N);
    }

    // Approach 6 — GemmKernel + V1 pipeline + DefaultGemm2DEpilogue (host-only)
    {
        GemmHostArgs host_args{a_dev, b_dev, d_dev,
                               1, M, N, K, stride_A, stride_B, stride_D};
        run_gemm_kernel<approach6::Kernel>(
            "Approach 6: GemmKernel + V1 + Default2DEpilogue",
            host_args, d_dev, h_d_ref, h_d, gflop, M, N);
    }

    // Approach 7 — GemmKernel + Mem pipeline + DefaultGemm2DEpilogue (host-only)
    {
        GemmHostArgs host_args{a_dev, b_dev, d_dev,
                               1, M, N, K, stride_A, stride_B, stride_D};
        run_gemm_kernel<approach7::Kernel>(
            "Approach 7: GemmKernel + Mem + Default2DEpilogue",
            host_args, d_dev, h_d_ref, h_d, gflop, M, N);
    }

    // Approach 8 — BlockGemm + double-buffered LDS (custom kernel)
    run_custom_kernel("Approach 8: BlockGemm + double-buffered LDS",
                      Approach8_BlockGemmLdsDouble{}, grid_size,
                      a_dev, b_dev, d_dev, h_d_ref, h_d, gflop, M, N, K);

    // Approach 9 — GemmKernel + CompV4 pipeline (DoubleSmemBuffer=true)
    {
        GemmHostArgs host_args{a_dev, b_dev, d_dev,
                               1, M, N, K, stride_A, stride_B, stride_D};
        run_gemm_kernel<approach9::Kernel>(
            "Approach 9: GemmKernel + CompV4 (double LDS buffer)",
            host_args, d_dev, h_d_ref, h_d, gflop, M, N);
    }

    std::cout << "============================================\n"
              << "Summary (lower ms = faster):\n"
              << "  1. BlockGemm (no LDS)            — manual everything\n"
              << "  2. BlockGemm + LDS               — B staged via LDS\n"
              << "  3. GemmPipeline (device kernel)   — mid-level, pipeline handles LDS+K-loop\n"
              << "  4. GemmKernel + V1 + CShuffle    — host-only, LDS shuffle epilogue\n"
              << "  5. GemmKernel + Mem + CShuffle   — host-only, LDS shuffle epilogue\n"
              << "  6. GemmKernel + V1 + Default2D   — host-only, direct-store epilogue\n"
              << "  7. GemmKernel + Mem + Default2D  — host-only, direct-store epilogue\n"
              << "  8. BlockGemm + 2x LDS            — double-buffered LDS ping-pong\n"
              << "  9. GemmKernel + CompV4            — double LDS buffer pipeline\n"
              << "============================================\n\n";

    return 0;
}
