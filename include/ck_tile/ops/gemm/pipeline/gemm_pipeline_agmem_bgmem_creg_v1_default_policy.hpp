// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// Default policy for GemmPipelineAGmemBGmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct GemmPipelineAGmemBGmemCRegV1DefaultPolicy
{
    template <index_t TileSizeMN, index_t TileSizeK, index_t BlockSize, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSize()
    {
        constexpr index_t ElemPerThread = (TileSizeMN * TileSizeK) / BlockSize;
        constexpr index_t MaxVectorSize = 16 / sizeof(DataType);
        static_assert(0 < ElemPerThread);

        return min(MaxVectorSize, ElemPerThread);
    }

#if 0
    // 2d
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
    
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t VectorSize = GetVectorSize<kMPerBlock, kKPerBlock, kBlockSize, typename Problem::ADataType>();

        constexpr auto a_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock), number<VectorSize>{});

        return a_lds_block_desc;
    }

    // 2d
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
    
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t VectorSize = GetVectorSize<kNPerBlock, kKPerBlock, kBlockSize, typename Problem::BDataType>();

        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), number<VectorSize>{});

        return b_lds_block_desc;
    }
#elif 1
    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t VectorSize =
            GetVectorSize<kMPerBlock, kKPerBlock, kBlockSize, typename Problem::ADataType>();

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(
                number<kKPerBlock / VectorSize>{}, number<kMPerBlock>{}, number<VectorSize>{}),
            make_tuple(number<(kMPerBlock + 1) * VectorSize>{}, number<VectorSize>{}, number<1>{}),
            number<VectorSize>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / VectorSize, VectorSize))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t VectorSize =
            GetVectorSize<kNPerBlock, kKPerBlock, kBlockSize, typename Problem::BDataType>();

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(
                number<kKPerBlock / VectorSize>{}, number<kNPerBlock>{}, number<VectorSize>{}),
            make_tuple(number<(kNPerBlock + 1) * VectorSize>{}, number<VectorSize>{}, number<1>{}),
            number<VectorSize>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / VectorSize, VectorSize))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }
#elif 1 // make sure we have added callee declarations before enabling codes below
    // fake XOR
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(number<kMPerBlock / 2>{}, number<2>{}, number<kKPerBlock>{}),
            number<kKPerBlock>{});

        constexpr index_t kK1 =
            GetVectorSize<kMPerBlock, kKPerBlock, kBlockSize, typename Problem::ADataType>();

        constexpr auto a_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            a_lds_block_desc_d1_d2_d3,
            make_tuple(
                make_xor_transform(make_tuple(number<kMPerBlock / 2>{}, number<kKPerBlock>{}), kK1),
                make_pass_through_transform(2)),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}));

        constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
            a_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(number<kMPerBlock / 2>{}, number<2>{})),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc_m_k;
    }

    // fake XOR
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(number<kNPerBlock / 2>{}, number<2>{}, number<kKPerBlock>{}),
            number<kKPerBlock>{});

        constexpr index_t kK1 =
            GetVectorSize<kNPerBlock, kKPerBlock, kBlockSize, typename Problem::BDataType>();

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(
                make_xor_transform(make_tuple(number<kNPerBlock / 2>{}, number<kKPerBlock>{}), kK1),
                make_pass_through_transform(2)),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(number<kNPerBlock / 2>{}, number<2>{})),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc_n_k;
    }
#endif
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t smem_size_b = sizeof(typename Problem::BDataType) *
                                        MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();
        index_t smem_size             = 0;
        smem_size += smem_size_a + smem_size_b;

        return smem_size;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        constexpr index_t MaxVectorSize = 16 / sizeof(ADataType);
        static_assert(0 < ElemPerThread);

        constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
#if 1 // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
#else // coalesce reading for each warps
        constexpr index_t M0 = kBlockSize / get_warp_size();
        constexpr index_t M1 = kMPerBlock / (M2 * M0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        constexpr index_t MaxVectorSize = 16 / sizeof(BDataType);
        static_assert(0 < ElemPerThread);

        constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
#if 1 // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        static_assert(N2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(N1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
#else // coalesce reading for each warps
        constexpr index_t N0 = kBlockSize / get_warp_size();
        constexpr index_t N1 = kNPerBlock / (N2 * N0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using AccDataType     = float;
        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                AccDataType,
                                                WarpTile::at(number<0>{}),
                                                WarpTile::at(number<1>{}),
                                                WarpTile::at(number<2>{}),
                                                /*TransposeC=*/true>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
