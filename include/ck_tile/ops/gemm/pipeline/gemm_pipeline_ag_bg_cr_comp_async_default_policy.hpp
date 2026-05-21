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
template <bool EnableSubTile = false>
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>;
    using Base::GetSmemPackA;
    using Base::GetSmemPackB;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::is_a_load_tr;
    using Base::is_b_load_tr;

    // Async copy supports 32-bit, 96-bit, or 128-bit transfers (4, 12, 16 bytes)
    template <typename DataType, index_t KPack>
    static constexpr bool IsSupportedAsyncVectorWidth =
        sizeof(DataType) * KPack == 4 || sizeof(DataType) * KPack == 12 ||
        sizeof(DataType) * KPack == 16;

    // XOR Swizzle: support FP8 / BF8
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleDataType =
        (std::is_same_v<remove_cvref_t<typename Problem::ADataType>, fp8_t> ||  // A FP8
         std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf8_t>) && // A BF8
        (std::is_same_v<remove_cvref_t<typename Problem::BDataType>, fp8_t> ||  // B FP8
         std::is_same_v<remove_cvref_t<typename Problem::BDataType>, bf8_t>);   // B BF8

    // Check that async vector store to LDS is supported
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleAsyncWidth =
        IsSupportedAsyncVectorWidth<typename Problem::ADataType,
                                    Base::template GetSmemPackA<Problem>()> &&
        IsSupportedAsyncVectorWidth<typename Problem::BDataType,
                                    Base::template GetSmemPackB<Problem>()>;

    // Assume normal LDS layout, not transpose-load
    template <typename Problem>
    static constexpr bool UseXorSwizzle =
        !Base::template is_a_load_tr<Problem> && !Base::template is_b_load_tr<Problem> &&
        IsSupportedXorSwizzleDataType<Problem> && IsSupportedXorSwizzleAsyncWidth<Problem>;

    // Compute the number of LDS read accesses for A or B
    // IsLoadTr=true if ds_read_tr is used
    template <bool IsLoadTr, typename DataType, index_t ThreadElements>
    CK_TILE_HOST_DEVICE static constexpr auto CalculateWGAttrNumAccess()
    {
        if constexpr(IsLoadTr)
        {
            // Transpose-load path: ds_read_tr reads DS_READ_TR_SIZE bytes per instruction.
            constexpr index_t vector_size =
                DS_READ_TR_SIZE() / sizeof(DataType) * numeric_traits<DataType>::PackedSize;
            if constexpr(vector_size == ThreadElements)
                return WGAttrNumAccessEnum::Single;
            else if constexpr(vector_size * 2 == ThreadElements)
                return WGAttrNumAccessEnum::Double;
            else if constexpr(vector_size * 4 == ThreadElements)
                return WGAttrNumAccessEnum::Quad;
            else
                return WGAttrNumAccessEnum::Invalid;
        }
        else
        {
            // Non-transpose path: ds_read_b128 reads 16 bytes per instruction
            constexpr index_t bytes_per_lane =
                sizeof(DataType) * ThreadElements / numeric_traits<DataType>::PackedSize;
            constexpr index_t ds_read_b128_width = 16;
            if constexpr(bytes_per_lane <= ds_read_b128_width)
                return WGAttrNumAccessEnum::Single;
            else if constexpr(bytes_per_lane <= ds_read_b128_width * 2)
                return WGAttrNumAccessEnum::Double;
            else if constexpr(bytes_per_lane <= ds_read_b128_width * 4)
                return WGAttrNumAccessEnum::Quad;
            else
                return WGAttrNumAccessEnum::Invalid;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAWGAttrNumAccess()
    {
        using WarpTile                    = typename Problem::BlockGemmShape::WarpTile;
        constexpr index_t thread_elements = WarpTile::at(I0) * WarpTile::at(I2) / get_warp_size();
        return CalculateWGAttrNumAccess<Base::template is_a_load_tr<Problem>,
                                        typename Problem::ADataType,
                                        thread_elements>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBWGAttrNumAccess()
    {
        using WarpTile                    = typename Problem::BlockGemmShape::WarpTile;
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        return CalculateWGAttrNumAccess<Base::template is_b_load_tr<Problem>,
                                        typename Problem::BDataType,
                                        thread_elements>();
    }

    // Get number of accesses
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWGAttrNumAccess()
    {
        constexpr auto num_access_a = GetAWGAttrNumAccess<Problem>();
        constexpr auto num_access_b = GetBWGAttrNumAccess<Problem>();

        if constexpr(num_access_a == WGAttrNumAccessEnum::Invalid ||
                     num_access_b == WGAttrNumAccessEnum::Invalid)
            return WGAttrNumAccessEnum::Invalid;
        else if constexpr(static_cast<index_t>(num_access_a) >= static_cast<index_t>(num_access_b))
            return num_access_a;
        else
            return num_access_b;
    }

    template <typename Problem,
              index_t MNPerBlock,
              index_t WarpTileMN,
              index_t K2,
              WGAttrNumAccessEnum WGAttrNumAccess>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXorSwizzledABLdsBlockDescriptor()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t KWarps    = BlockWarps::at(I2);
        constexpr index_t K1        = WarpTile::at(I2) / K2;
        constexpr index_t K0        = KPerBlock / (KWarps * K1 * K2);

        constexpr index_t warp_size            = get_warp_size();
        constexpr index_t warp_num             = BlockSize / warp_size;
        constexpr index_t wg_attr_num_access_v = static_cast<index_t>(WGAttrNumAccess);

        static_assert(warp_num * warp_size == BlockSize, "Wrong!");
        static_assert(KWarps * K0 * K1 * K2 == KPerBlock, "Wrong!");
        static_assert(WGAttrNumAccess != WGAttrNumAccessEnum::Invalid,
                      "XOR swizzle: unsupported LDS read access count for this configuration");

        constexpr index_t M4 = warp_size / wg_attr_num_access_v / K1;
        constexpr index_t M3 = wg_attr_num_access_v;
        constexpr index_t M2 = WarpTileMN / M4 / M3;
        constexpr index_t M1 = (warp_num / Problem::NumWaveGroups) / M2;
        constexpr index_t M0 = MNPerBlock / M1 / M2 / M3 / M4;

        static_assert(M1 * M0 * M2 * M3 * M4 == MNPerBlock, "Wrong!");

        constexpr index_t PadSize = 16;

        constexpr auto desc_0 = make_naive_tensor_descriptor(
            number_tuple<M2, KWarps, M1, M0, K0, M3, M4, K1, K2>{},
            number_tuple<KWarps * M1 * M0 * K0 * M3 * M4 * K1 * K2 + PadSize,
                         M1 * M0 * K0 * M3 * M4 * K1 * K2,
                         M0 * K0 * M3 * M4 * K1 * K2,
                         K0 * M3 * M4 * K1 * K2,
                         M3 * M4 * K1 * K2,
                         M4 * K1 * K2,
                         K1 * K2,
                         K2,
                         1>{},
            number<K2>{},
            number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M2>{}),
                       make_pass_through_transform(number<KWarps>{}),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M3>{}),
                       make_xor_transform(make_tuple(number<M4>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4>{},
                       sequence<5>{},
                       sequence<6, 7>{},
                       sequence<8>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4>{},
                       sequence<5>{},
                       sequence<6, 7>{},
                       sequence<8>{}));

        constexpr auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M3, M4>{}),
                       make_merge_transform_v3_division_mod(number_tuple<KWarps, K0, K1, K2>{})),
            make_tuple(sequence<3, 2, 0, 5, 6>{}, sequence<1, 4, 7, 8>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_2;
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeALdsBlockDescriptor<Problem>();
#else
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_a_load_tr<Problem>)
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
            if constexpr(UseXorSwizzle<Problem>)
            {
                using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
                constexpr index_t KPack = Base::template GetSmemPackA<Problem>();
                constexpr auto desc =
                    MakeXorSwizzledABLdsBlockDescriptor<Problem,
                                                        MPerBlock,
                                                        WarpTile::at(I0),
                                                        KPack,
                                                        GetWGAttrNumAccess<Problem>()>();
                static_assert(desc.get_element_space_size() >= MPerBlock * KPerBlock,
                              "XOR swizzle LDS allocation must cover the A tile");
                return desc;
            }
            else
            {
                constexpr index_t KPack = Base::template GetSmemPackA<Problem>();

                constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                return transform_tensor_descriptor(
                    a_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<MPerBlock>{}),
                               make_merge_transform(
                                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeBLdsBlockDescriptor<Problem>();
#else
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_b_load_tr<Problem>)
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
            if constexpr(UseXorSwizzle<Problem>)
            {
                using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
                constexpr index_t KPack = Base::template GetSmemPackB<Problem>();
                constexpr auto desc =
                    MakeXorSwizzledABLdsBlockDescriptor<Problem,
                                                        NPerBlock,
                                                        WarpTile::at(I1),
                                                        KPack,
                                                        GetWGAttrNumAccess<Problem>()>();
                static_assert(desc.get_element_space_size() >= NPerBlock * KPerBlock,
                              "XOR swizzle LDS allocation must cover the B tile");
                return desc;
            }
            else
            {
                constexpr index_t KPack = Base::template GetSmemPackB<Problem>();

                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KPack>{}, number<NPerBlock>{}, number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                return transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<NPerBlock>{}),
                               make_merge_transform(
                                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
#endif
    }

    template <typename Problem, index_t K2, WGAttrNumAccessEnum WGAttrNumAccess, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadABDramWindow(const Window& window)
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr auto ndims = std::decay_t<decltype(window)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");

        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t KWarps    = BlockWarps::at(I2);
        constexpr index_t K1        = WarpTile::at(I2) / K2;

        static_assert(K1 * K2 == WarpTile::at(I2), "Wrong!");
        static_assert(KPerBlock % (KWarps * K1 * K2) == 0, "Wrong!");

        constexpr index_t wg_attr_num_access_v = static_cast<index_t>(WGAttrNumAccess);

        constexpr index_t M4 = get_warp_size() / wg_attr_num_access_v / K1;
        static_assert(get_warp_size() % (wg_attr_num_access_v * K1) == 0,
                      "warp_size must be divisible by (wg_attr_num_access_v * K1)");

        auto&& tensor_view      = window.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view.get_tensor_descriptor().get_lengths();

        const index_t k_tiles = cols / (KWarps * K1 * K2);
        const auto col_lens   = make_tuple(k_tiles, number<KWarps>{}, number<K1>{}, number<K2>{});

        const index_t M0    = integer_divide_ceil(rows, M4);
        const auto row_lens = make_tuple(M0, number<M4>{});

        const auto desc_0 = transform_tensor_descriptor(
            tensor_view.get_tensor_descriptor(),
            make_tuple(make_unmerge_transform(row_lens), make_unmerge_transform(col_lens)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}));

        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M4>{}, number<K1>{})),
                       make_pass_through_transform(k_tiles),
                       make_pass_through_transform(number<KWarps>{}),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}));

        const auto desc =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tile_window(
            make_tensor_view<address_space_enum::global>(&tensor_view.get_buffer_view()(0), desc),
            window.get_window_lengths(),
            window.get_window_origin());
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const Window& window)
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t KPack = Base::template GetSmemPackA<Problem>();
            return MakeAsyncLoadABDramWindow<Problem, KPack, GetWGAttrNumAccess<Problem>()>(window);
        }
        else
        {
            return make_tile_window(window.get_bottom_tensor_view(),
                                    window.get_window_lengths(),
                                    window.get_window_origin());
        }
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadBDramWindow(const Window& window)
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t KPack = Base::template GetSmemPackB<Problem>();
            return MakeAsyncLoadABDramWindow<Problem, KPack, GetWGAttrNumAccess<Problem>()>(window);
        }
        else
        {
            return make_tile_window(window.get_bottom_tensor_view(),
                                    window.get_window_lengths(),
                                    window.get_window_origin());
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetEstimatedVgprCount()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I1);
        constexpr index_t warpSize     = get_warp_size();
        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t AccVGPRNum =
            sizeof(CDataType) * MPerBlock * NPerBlock / BlockSize / BytesPerVGPR;

        constexpr index_t DoubleBufferFactor = 3;

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

        constexpr index_t ALoadVGPRNum = sizeof(ADataType) / APackedSize * MPerBlock * KPerBlock /
                                         MWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t BLoadVGPRNum = sizeof(BDataType) / BPackedSize * NPerBlock * KPerBlock /
                                         NWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t TotalInputVGPRNum = ALoadVGPRNum + BLoadVGPRNum;

        return make_tuple(number<AccVGPRNum>{}, number<TotalInputVGPRNum>{});
    }

    // this function is used to get SubTile Number
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr auto estimated_vgpr = GetEstimatedVgprCount<Problem>();

        constexpr auto acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr auto input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_capacity = get_max_vgpr_count();
        // sub tile number; have 1, 2, 4 choices
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_capacity) ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_capacity)
                                             ? 2
                                             : 4;

        return number<sub_tile_num>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr auto wg_attr_num_access = GetWGAttrNumAccess<Problem>();

        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = EnableSubTile ? pipeline_tune_params.value : 1;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
