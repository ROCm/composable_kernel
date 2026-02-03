// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "block_gemm_asmem_bsmem_creg.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// Policy for BlockGemmPipelineAGmemBGmemCReg with XOR-based bank-conflict-free optimization
struct BlockGemmPipelineAGmemBGmemCRegPolicy
{
    // XOR-based bank-conflict-free LDS layout for A
    // This optimization uses XOR transformations to avoid bank conflicts in shared memory
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        // Create initial 3D descriptor with layering
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                       number<kMPerBlock / MLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        // Apply XOR transform to permute indices and avoid bank conflicts
        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                     number<kKPerBlock / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        // Unmerge and rearrange dimensions
        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        // Final merge to get [M, K] layout
        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    // XOR-based bank-conflict-free LDS layout for B
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

        constexpr auto DataTypeSize = sizeof(BDataType);
        constexpr auto NLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        // Create initial 3D descriptor with layering
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                       number<kNPerBlock / NLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * NLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        // Apply XOR transform to permute indices and avoid bank conflicts
        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                     number<kKPerBlock / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        // Unmerge and rearrange dimensions
        constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        // Final merge to get [N, K] layout
        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(ADataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmASmemBSmemCReg<Problem>{};
    }
};

} // namespace ck_tile
