// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

// C0 = A0 * B0
// D0 = softmax(C0)
// C1 = D0 * B1
template <typename A0DataType,
          typename B0DataType,
          typename Acc0DataType,
          typename C0DataType,
          typename B1DataType,
          typename Acc1DataType,
          typename C1DataType,
          ck::index_t kBlockSize,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmSoftmaxGemm
{
    // block gemm0 pipeline
    using BlockGemm0Pipeline = ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<
        ck::tile_program::block::BlockGemmPipelineProblem<
            A0DataType,
            B0DataType,
            Acc0DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>,
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>;

    // block gemm1
    using BlockGemm1 = ck::tile_program::block::BlockGemmARegBSmemCRegV1<
        ck::tile_program::block::BlockGemmARegBSmemCRegV1Problem<
            C0DataType,
            B1DataType,
            Acc1DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN1PerBlock, kN0PerBlock>>,
        ck::tile_program::block::BlockGemmARegBSmemCRegV1DefaultPolicy>;

#if 0
    // 2d
    __host__ __device__ static constexpr auto MakeB1LdsBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        return b_lds_block_desc;
    }
#else
    // fake XOR
    __host__ __device__ static constexpr auto MakeB1LdsBlockDescriptor()
    {
        using namespace ck;

        using BDataType = B1DataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kNPerBlock / 2, 2, kKPerBlock), Number<kKPerBlock>{});

        constexpr index_t kK1 = 16 / sizeof(BDataType);

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kNPerBlock / 2, kKPerBlock), kK1),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kNPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc_n_k;
    }
#endif

    __host__ __device__ static constexpr auto MakeB1DramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        using BDataType = B1DataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        using namespace ck;

        return math::max(BlockGemm0Pipeline::GetStaticLdsSize(),
                         static_cast<index_t>(MakeB1LdsBlockDescriptor().GetElementSpaceSize() *
                                              sizeof(B1DataType)));
    }

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const A0DataType* p_a0,
                                        const B0DataType* p_b0,
                                        const B1DataType* p_b1,
                                        C1DataType* p_c1,
                                        ck::index_t M0,
                                        ck::index_t N0,
                                        ck::index_t K0,
                                        ck::index_t N1,
                                        ck::index_t Lda0,
                                        ck::index_t Ldb0,
                                        ck::index_t Ldb1,
                                        ck::index_t Ldc1)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = ps.read_first_lane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = ps.read_first_lane(id_tile.At<1>() * kN1PerBlock);

        __shared__ char p_smem_char[GetStaticLdsSize()];

        // A0 DRAM block window
        auto a0_dram_block_window = make_tile_window(
            a0_dram_grid, make_tuple(Number<kM0PerBlock>{}, Number<kK0PerBlock>{}), {iM0, 0});

        // B0 DRAM block window
        auto b0_dram_block_window = make_tile_window(
            b0_dram_grid, make_tuple(Number<kN0PerBlock>{}, Number<kK0PerBlock>{}), {0, 0});

        // Block GEMM0 pipeline
        constexpr auto block_gemm0_pipeline = BlockGemm0Pipeline{};

        // B1 DRAM window
        auto b1_dram_block_window =
            make_tile_window(b1_dram_grid,
                             make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}),
                             {iN1, 0},
                             MakeB1DramTileDistribution());

        // B1 LDS tensor view: occupies the same LDS allocation as block_gemm0_pipeline
        auto b1_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<B1DataType*>(p_smem_char), MakeB1LdsBlockDescriptor());

        auto b1_lds_block_window = make_tile_window(
            b1_lds_block, make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}), {0, 0});

        // Bock GEMM1
        constexpr auto block_gemm1 = BlockGemm1{};

        // Acc0 tile
        using Acc0BlockTileType =
            decltype(block_gemm0_pipeline(a0_dram_block_window, b0_dram_block_window, 0, nullptr));

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm1(
            tile_elementwise_in(type_convert<C0DataType, Acc0DataType>, Acc0BlockTileType{}),
            b1_dram_block_window)){};

        const auto f_max = [](auto v0, auto v1) { return max(v0, v1); };
        const auto f_sum = [](auto v0, auto v1) { return v0 + v1; };

        // init Acc1
        tile_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        // m, l tile
        auto m = decltype(block_tile_reduce<Acc0DataType>(
            Acc0BlockTileType{}, Sequence<1>{}, f_max, Acc0DataType{0})){};

        // init m, l
        auto l = make_static_distributed_tensor<Acc0DataType>(m.GetTileDistribution());

        tile_elementwise_inout([](auto& m_v) { m_v = NumericLimits<Acc0DataType>::Lowest(); }, m);
        tile_elementwise_inout([](auto& l_v) { l_v = 0; }, l);

        index_t iN0 = 0;

        do
        {
            // S[i][j] = Q[i] * K[j]
            const auto acc0_block_tile = block_gemm0_pipeline(
                a0_dram_block_window, b0_dram_block_window, K0 / kK0PerBlock, p_smem_char);

            // rowmax(S[i][j])
            auto m_local = block_tile_reduce<Acc0DataType>(
                acc0_block_tile, Sequence<1>{}, f_max, NumericLimits<Acc0DataType>::Lowest());

            block_tile_reduce_sync(m_local, f_max);

            // m[i][j-1]
            const auto m_old = m;

            // m[i][j]
            tile_elementwise_inout(
                [](auto& m_v, auto m_old_v, auto m_local_v) { m_v = max(m_old_v, m_local_v); },
                m,
                m_old,
                m_local);

            // P[i][j]
            auto p =
                make_static_distributed_tensor<Acc0DataType>(acc0_block_tile.GetTileDistribution());

            constexpr auto p_spans = decltype(p)::GetDistributedSpans();

            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_v = m.GetElementFromTileDistributedIndices(i_idx);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    const auto s_v = acc0_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

                    const auto p_v = math::exp(s_v - m_v);

                    p.SetElementFromTileDistributedIndices(i_j_idx, p_v);
                });
            });

            // rowsum(P[i][j])
            auto rowsum_p =
                block_tile_reduce<Acc0DataType>(p, Sequence<1>{}, f_sum, Acc0DataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum);

            // l[i][j], O[i][j]
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_old_v = m_old.GetElementFromTileDistributedIndices(i_idx);
                const auto m_v     = m.GetElementFromTileDistributedIndices(i_idx);
                const auto l_old_v = l.GetElementFromTileDistributedIndices(i_idx);

                const auto tmp  = math::exp(m_old_v - m_v);
                const auto tmp2 = 1 / tmp;

                auto l_v = tmp * l_old_v + rowsum_p.GetElementFromTileDistributedIndices(i_idx);

                l.SetElementFromTileDistributedIndices(i_idx, l_v);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    // O[i][j]
                    const auto o_old_v =
                        acc1_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

#if 0 // debug
      // this use the same equation from FA v2 paper, but produce -nan
                    const auto o_v = o_old_v * tmp2;
#elif 1
                    // this use different equation from FA v2 paper, but produce correct result
                    (void) tmp2;
                    const auto o_v = o_old_v * tmp;
#endif

                    acc1_block_tile.SetElementFromTileDistributedIndices(i_j_idx, o_v);
                });
            });

            // type cast p into a1
            const auto c0_block_tile =
                tile_elementwise_in(type_convert<C0DataType, Acc0DataType>, p);

            // Block GEMM1: acc1 += c0 * b1
            {
                // load b1
                const auto b1_block_tile = load_tile(b1_dram_block_window);

                // wait for block gemm0 pipeline to finish
                ps.block_sync_lds();

                store_tile(b1_lds_block_window, b1_block_tile);

                // wait for store_tile to finish
                ps.block_sync_lds();

                // acc1 += c0 * b1
                block_gemm1(acc1_block_tile, c0_block_tile, b1_lds_block_window);

                // wait for block gemm1 to finish
                ps.block_sync_lds();
            }

            // move tile windows
            move_tile_window(b0_dram_block_window, {kN0PerBlock, 0});
            move_tile_window(b1_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // o[i][J-1]
        constexpr auto o_spans = decltype(acc1_block_tile)::GetDistributedSpans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            const auto l_v = l.GetElementFromTileDistributedIndices(i_idx);

            const auto tmp = 1 / l_v;

            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                const auto o_v = acc1_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

                const auto o_new_v = o_v * tmp;

                acc1_block_tile.SetElementFromTileDistributedIndices(i_j_idx, o_new_v);
            });
        });

        // type cast acc1 into c1
        const auto c1_block_tile =
            tile_elementwise_in(type_convert<C1DataType, Acc1DataType>, acc1_block_tile);

        // store c1
        auto c1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c1, make_tuple(M0, N1), make_tuple(Ldc1, 1), Number<32>{}, Number<1>{});

        auto c1_dram_window =
            make_tile_window(c1_dram_grid,
                             make_tuple(Number<kM0PerBlock>{}, Number<kN1PerBlock>{}),
                             {iM0, iN1},
                             c1_block_tile.GetTileDistribution());

        store_tile(c1_dram_window, c1_block_tile);
    }
};
