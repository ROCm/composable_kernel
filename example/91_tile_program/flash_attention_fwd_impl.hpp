// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/slice_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

// S[M0, N0] = Q[M0, K0] * K[N0, K0]
// P[M0, N0] = Softmax(S[M0, N0])
// O[M0, N1] = P[M0, N0] * V[N1, N0]
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType,
          ck::index_t kBlockSize,
          ck::index_t kHeadDim,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock,
          ck::index_t kK1PerBlock>
struct FlashAttentionFwdImpl
{
    // block gemm0 pipeline
    using BlockGemm0Problem = ck::tile_program::block::BlockGemmPipelineProblem<
        QDataType,
        KDataType,
        SaccDataType,
        kBlockSize,
        ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>;

    using BlockGemm0Policy =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPersistentQRegCachePolicy<
            kHeadDim>;

    using BlockGemm0Pipeline =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<BlockGemm0Problem,
                                                                   BlockGemm0Policy>;

    // block gemm1
    using BlockGemm1 = ck::tile_program::block::BlockGemmARegBSmemCRegV1<
        ck::tile_program::block::BlockGemmARegBSmemCRegProblem<
            PDataType,
            VDataType,
            OaccDataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN1PerBlock, kK1PerBlock>>,
        ck::tile_program::block::BlockGemmARegBSmemCRegV1DefaultPolicy>;

    // 3d, with padding
    __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        using namespace ck;

        // using BDataType = B1DataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kK1PerBlock;
        constexpr index_t kPad       = 1;
        // 2% faster than use kK1 = 8
        constexpr index_t kK1 = 4;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kK1>{}, Number<kNPerBlock>{}, Number<kK1>{}),
            make_tuple(Number<(kNPerBlock + kPad) * kK1>{}, Number<kK1>{}, Number<1>{}),
            Number<kK1>{},
            Number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(Number<kKPerBlock / kK1>{}, Number<kK1>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc;
    }

    __device__ static constexpr auto MakeVDramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        using BDataType = VDataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kK1PerBlock;

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

    __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        using namespace ck;

        return math::max(BlockGemm0Pipeline::GetStaticLdsSize(),
                         static_cast<index_t>(MakeVLdsBlockDescriptor().GetElementSpaceSize() *
                                              sizeof(VDataType)));
    }

    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const ck::index_t M0,
                               const ck::index_t N0,
                               const ck::index_t K0,
                               const ck::index_t N1,
                               const ck::index_t StrideQ,
                               const ck::index_t StrideK,
                               const ck::index_t StrideV,
                               const ck::index_t StrideO,
                               const ck::index_t iM0,
                               const ck::index_t iN1) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // allocate LDS
        __shared__ char smem_ptr[GetStaticLdsSize()];

        // Q/K/V DRAM and DRAM window
        const auto q_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr, make_tuple(M0, K0), make_tuple(StrideQ, 1), Number<32>{}, Number<1>{});

        const auto k_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr, make_tuple(N0, K0), make_tuple(StrideK, 1), Number<32>{}, Number<1>{});

        const auto v_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            v_ptr, make_tuple(N1, N0), make_tuple(StrideV, 1), Number<32>{}, Number<1>{});

        auto q_dram_window = make_tile_window(
            q_dram, make_tuple(Number<kM0PerBlock>{}, Number<kK0PerBlock>{}), {iM0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(Number<kN0PerBlock>{}, Number<kK0PerBlock>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(Number<kN1PerBlock>{}, Number<kK1PerBlock>{}),
                             {iN1, 0},
                             MakeVDramTileDistribution());

        // Q in Register
        auto q_reg_tensor = make_static_distributed_tensor<QDataType>(
            BlockGemm0Policy::template MakeARegBlockDescriptor<BlockGemm0Problem>());

        // V LDS and LDS window
        // V LDS occupies the same LDS allocation Q/K LDS
        auto v_lds = make_tensor_view<AddressSpaceEnum::Lds>(reinterpret_cast<VDataType*>(smem_ptr),
                                                             MakeVLdsBlockDescriptor());

        auto v_lds_window = make_tile_window(
            v_lds, make_tuple(Number<kN1PerBlock>{}, Number<kK1PerBlock>{}), {0, 0});

        // Block GEMM0 pipeline and Block GEMM1
        constexpr auto gemm0_pipeline = BlockGemm0Pipeline{};
        constexpr auto gemm1          = BlockGemm1{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SaccBlockTileType =
            decltype(gemm0_pipeline(q_dram_window, k_dram_window, q_reg_tensor, nullptr));

        using SBlockTileType = decltype(tile_elementwise_in(
            type_convert<SMPLComputeDataType, SaccDataType>, SaccBlockTileType{}));

        using PBlockTileType = decltype(tile_elementwise_in(type_convert<PDataType, SaccDataType>,
                                                            SaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, Sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm1(
            get_slice_tile(
                PBlockTileType{}, Sequence<0, 0>{}, Sequence<kM0PerBlock, kK1PerBlock>{}),
            v_dram_window));

        // init Sacc, Oacc, M, L
        auto s_acc = SaccBlockTileType{};
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        tile_elementwise_inout([](auto& e) { e = 0; }, o_acc);
        tile_elementwise_inout([](auto& e) { e = NumericLimits<SMPLComputeDataType>::Lowest(); },
                               m);
        tile_elementwise_inout([](auto& e) { e = 0; }, l);

        // loop over Column of S (J loop)
        index_t iN0 = 0;

        // Cold Q_Reg_Cache
        s_acc = gemm0_pipeline(q_dram_window, k_dram_window, q_reg_tensor, smem_ptr);
        do
        {
            // Hot Q_Reg_Cache
            if(iN0 > 0)
            {
                s_acc = gemm0_pipeline(k_dram_window, q_reg_tensor, smem_ptr);
            }
            // S{j}
            const auto s =
                tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc);

            // prefetch load v tile
            const auto v_prefetch = load_tile(v_dram_window);

            // m_local = rowmax(S{j})
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s, Sequence<1>{}, f_max, NumericLimits<SMPLComputeDataType>::Lowest());

            block_tile_reduce_sync(m_local, f_max);

            // m{j-1}
            const auto m_old = m;

            // m{j}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

            // Pcompute{j}
            auto p_compute =
                make_static_distributed_tensor<SMPLComputeDataType>(s.GetTileDistribution());

            constexpr auto p_spans = decltype(p_compute)::GetDistributedSpans();

            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    p_compute(i_j_idx) = math::exp(s[i_j_idx] - m[i_idx]);
                });
            });

            // rowsum(Pcompute{j})
            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, Sequence<1>{}, f_sum, SMPLComputeDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum);

            // l{j}, Oacc{j}
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto tmp = math::exp(m_old[i_idx] - m[i_idx]);

                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds();
            store_tile(v_lds_window, v_prefetch);
            move_tile_window(v_dram_window, {0, kK1PerBlock});

            // type cast Pcompute{j} into P{j}
            const auto p =
                tile_elementwise_in(type_convert<PDataType, SMPLComputeDataType>, p_compute);

            // Oacc{j}
            constexpr index_t k1_loops = kN0PerBlock / kK1PerBlock;

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    gemm1(o_acc,
                          get_slice_tile(p,
                                         Sequence<0, i_k1 * kK1PerBlock>{},
                                         Sequence<kM0PerBlock, (i_k1 + 1) * kK1PerBlock>{}),
                          v_lds_window);
                    block_sync_lds();
                    store_tile(v_lds_window, v);
                    move_tile_window(v_dram_window, {0, kK1PerBlock});
                });
            }
            // tail
            {
                block_sync_lds();
                gemm1(o_acc,
                      get_slice_tile(p,
                                     Sequence<0, (k1_loops - 1) * kK1PerBlock>{},
                                     Sequence<kM0PerBlock, kN0PerBlock>{}),
                      v_lds_window);
                block_sync_lds();
            }
            // move tile windows
            move_tile_window(k_dram_window, {kN0PerBlock, 0});
            iN0 += kN0PerBlock;
        } while(iN0 < N0);

        // Oacc
        constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            const auto tmp = 1 / l[i_idx];

            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                o_acc(i_j_idx) *= tmp;
            });
        });

        // type cast Oacc into O
        const auto o = tile_elementwise_in(type_convert<ODataType, OaccDataType>, o_acc);

        // O DRAM and O DRAM window
        auto o_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            o_ptr, make_tuple(M0, N1), make_tuple(StrideO, 1), Number<32>{}, Number<1>{});

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(Number<kM0PerBlock>{}, Number<kN1PerBlock>{}),
                             {iM0, iN1},
                             o.GetTileDistribution());

        // store O
        store_tile(o_dram_window, o);
    }
};
