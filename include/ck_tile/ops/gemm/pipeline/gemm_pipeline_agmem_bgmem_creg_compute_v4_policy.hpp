// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV4, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAGmemBGmemCregComputeV4DefaultPolicy : public UniversalGemmBasePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using BlockGemm = remove_cvref_t<decltype(GetBlockGemm<Problem>())>;
        using WG        = typename BlockGemm::WarpGemm;

        constexpr bool TransposeC = Problem::TransposeC;
        using CLayout             = typename Problem::CLayout;
        using CWarpDstr           = typename WG::CWarpDstr;

        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(TransposeC)
            {
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(TransposeC)
            {
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
            else
            {
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        using BlockGemm         = decltype(GetBlockGemm<Problem>());
        constexpr index_t KPack = BlockGemm::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackB()
    {
        using BlockGemm         = decltype(GetBlockGemm<Problem>());
        constexpr index_t KPack = BlockGemm::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack      = GetSmemPackA<Problem>();

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / KPack>{}, number<kMPerBlock>{}, number<KPack>{}),
            make_tuple(number<kMPerBlock * KPack>{}, number<KPack>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock>{} / KPack, number<KPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack      = GetSmemPackB<Problem>();

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / KPack>{}, number<kNPerBlock>{}, number<KPack>{}),
            make_tuple(number<(kNPerBlock)*KPack>{}, number<KPack>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a =
            integer_least_multiple(sizeof(typename Problem::ADataType) *
                                       MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                                   16);
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t smem_size_b =
            integer_least_multiple(sizeof(typename Problem::BDataType) *
                                       MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
                                   16);
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();

        return smem_size_a + smem_size_b;
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
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
