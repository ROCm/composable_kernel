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
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"

// C1 = A0 * B0 * B1
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
struct GemmGemm
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
    __device__ static constexpr auto MakeB1LdsBlockDescriptor()
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
    __device__ static constexpr auto MakeB1LdsBlockDescriptor()
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

    __device__ static constexpr auto MakeB1DramTileDistribution()
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

    __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        using namespace ck;

        return math::max(BlockGemm0Pipeline::GetStaticLdsSize(),
                         static_cast<index_t>(MakeB1LdsBlockDescriptor().GetElementSpaceSize() *
                                              sizeof(B1DataType)));
    }

    __device__ void operator()(const A0DataType* p_a0,
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

        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = get_block_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = __builtin_amdgcn_readfirstlane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = __builtin_amdgcn_readfirstlane(id_tile.At<1>() * kN1PerBlock);

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

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm1(
            tile_elementwise_in(
                type_convert<C0DataType, Acc0DataType>,
                block_gemm0_pipeline(a0_dram_block_window, b0_dram_block_window, 0, nullptr)),
            b1_dram_block_window)){};

        // init Acc1
        tile_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        index_t iN0 = 0;

        do
        {
            // Block GEMM0 pipeline: acc0 = a0 * b0
            const auto acc0_block_tile = block_gemm0_pipeline(
                a0_dram_block_window, b0_dram_block_window, K0 / kK0PerBlock, p_smem_char);

            // type cast acc0 into c0
            const auto c0_block_tile =
                tile_elementwise_in(type_convert<C0DataType, Acc0DataType>, acc0_block_tile);

            // Block GEMM1: acc1 += c0 * b1
            {
                // load b1
                const auto b1_block_tile = load_tile(b1_dram_block_window);

                // wait for block gemm0 pipeline to finish
                block_sync_lds();

                store_tile(b1_lds_block_window, b1_block_tile);

                // wait for store_tile to finish
                block_sync_lds();

                // acc1 += c0 * b1
                block_gemm1(acc1_block_tile, c0_block_tile, b1_lds_block_window);

                // wait for block gemm1 to finish
                block_sync_lds();
            }

            // move tile windows
            move_tile_window(b0_dram_block_window, {kN0PerBlock, 0});
            move_tile_window(b1_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

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
