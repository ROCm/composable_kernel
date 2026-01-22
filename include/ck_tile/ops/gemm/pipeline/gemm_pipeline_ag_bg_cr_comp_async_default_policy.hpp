// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t DWORDx4            = 16;

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeALdsBlockDescriptor
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackA<Problem>();

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<MPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeBLdsBlockDescriptor
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackB<Problem>();

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<NPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<NPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // Methods for async byte-based loading (similar to mx_flatmm)
    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using ADataType               = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        constexpr index_t K2 = DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4;
        const index_t K0     = cols / (K1 * K2 * APackedSize);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t M1 = 4;
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t M2_dstr   = WaveSize / K1;
        constexpr index_t M1_dstr   = BlockSize / WaveSize;
        constexpr index_t M0_dstr   = MPerBlock / (M2_dstr * M1_dstr);

        constexpr auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0_dstr, M1_dstr, M2_dstr>, sequence<K0, K1, K2>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 2>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<MPerBlock>{}, number<KPerBlock / APackedSize>{}),
                                {origin_tmp[0], origin_tmp[1] / APackedSize},
                                tile_dstr);
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeBAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using BDataType               = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        constexpr index_t K2 = DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4;
        const index_t K0     = cols / (K1 * K2 * BPackedSize);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t N1 = 4;
        const index_t N0     = integer_divide_ceil(rows, N1);
        const auto row_lens  = make_tuple(N0, number<N1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(N0),
                       make_xor_transform(make_tuple(number<N1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t N2_dstr   = WaveSize / K1;
        constexpr index_t N1_dstr   = BlockSize / WaveSize;
        constexpr index_t N0_dstr   = NPerBlock / (N2_dstr * N1_dstr);

        constexpr auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0_dstr, N1_dstr, N2_dstr>, sequence<K0, K1, K2>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 2>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<NPerBlock>{}, number<KPerBlock / BPackedSize>{}),
                                {origin_tmp[0], origin_tmp[1] / BPackedSize},
                                tile_dstr);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
