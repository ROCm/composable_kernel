// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "block_gemm_asmem_bsmem_creg.hpp"

#include "config.h"

namespace ck_tile {

// Default policy for BlockGemmPipelineAGmemBGmemCReg
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmPipelineAGmemBGmemCRegDefaultPolicy
{
    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

#if defined(NAIVE_IMPLEMENTATION)
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(PADDING_K_FIRST)
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<(kKPerBlock / kKPack + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(PADDING_MN_FIRST)
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(USING_XOR_BASED_BANK_CONFLICT_FREE)
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                    number<kMPerBlock / MLdsLayer>{},
                    number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                    number<kKPerBlock / kKPack * MLdsLayer>{})),
                    make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                        make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                        make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                        make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform(
                        make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                        make_merge_transform(
                        make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
#endif

        return a_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

#if defined(PADDING_K_FIRST) || defined(NAIVE_IMPLEMENTATION)
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(PADDING_K_FIRST)
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<(kKPerBlock / kKPack + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(PADDING_MN_FIRST)
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

#elif defined(USING_XOR_BASED_BANK_CONFLICT_FREE)
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr auto DataTypeSize = sizeof(BDataType);
        constexpr auto NLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                    number<kNPerBlock / NLdsLayer>{},
                    number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * NLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                    number<kKPerBlock / kKPack * NLdsLayer>{})),
                    make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                        make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                        make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                        make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform(
                        make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                        make_merge_transform(
                        make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
#endif

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
