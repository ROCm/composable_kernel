// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_mx/block/block_mx_gemm_areg_breg_creg_v1.hpp"
#include <type_traits>

namespace ck_tile {
// Default policy for MXGemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
// Adds MX scale tile distributions
struct MXGemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    // Async copy supports 32-bit, 96-bit, or 128-bit transfers (4, 12, 16 bytes)
    // Take PackedSize into consideration (for example for FP4 support)
    template <typename DataType, index_t KPack>
    static constexpr index_t AsyncVectorBytes =
        sizeof(DataType) * KPack / numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    template <typename DataType, index_t KPack>
    static constexpr bool IsSupportedAsyncVectorWidth =
        AsyncVectorBytes<DataType, KPack> == 4 || AsyncVectorBytes<DataType, KPack> == 12 ||
        AsyncVectorBytes<DataType, KPack> == 16;

    template <typename DataType>
    static constexpr bool IsF8XorSwizzleDataType =
        std::is_same_v<remove_cvref_t<DataType>, fp8_t> ||
        std::is_same_v<remove_cvref_t<DataType>, bf8_t>;

    template <typename DataType>
    static constexpr bool IsFP4XorSwizzleDataType =
        std::is_same_v<remove_cvref_t<DataType>, pk_fp4_t>;

    // XOR Swizzle: support F8/F8 and FP4/FP4. Mixed F8/FP4 stays on the plain path.
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleDataType =
        (IsF8XorSwizzleDataType<typename Problem::ADataType> &&
         IsF8XorSwizzleDataType<typename Problem::BDataType>) ||
        (IsFP4XorSwizzleDataType<typename Problem::ADataType> &&
         IsFP4XorSwizzleDataType<typename Problem::BDataType>);

    // FP4 needs the XOR KPack in logical elements
    // so the async transaction remains 16 bytes
    template <typename DataType, index_t SmemPack>
    static constexpr index_t GetXorSwizzleKPack()
    {
        return SmemPack * numeric_traits<remove_cvref_t<DataType>>::PackedSize;
    }

    template <typename Problem>
    static constexpr index_t GetXorSwizzleKPackA()
    {
        return GetXorSwizzleKPack<typename Problem::ADataType, GetSmemPackA<Problem>()>();
    }

    template <typename Problem>
    static constexpr index_t GetXorSwizzleKPackB()
    {
        return GetXorSwizzleKPack<typename Problem::BDataType, GetSmemPackB<Problem>()>();
    }

    // Check that async vector store to LDS is supported
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleAsyncWidth =
        IsSupportedAsyncVectorWidth<typename Problem::ADataType, GetXorSwizzleKPackA<Problem>()> &&
        IsSupportedAsyncVectorWidth<typename Problem::BDataType, GetXorSwizzleKPackB<Problem>()>;

    // gfx950 scales:16x16x128 warp tile, 16-element smem pack, KWarps==1
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleShape = []() {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        return Problem::NumWaveGroups == 1 && BlockWarps::at(number<2>{}) == 1 &&
               WarpTile::at(number<0>{}) == 16 && WarpTile::at(number<1>{}) == 16 &&
               WarpTile::at(number<2>{}) == 128 && GetSmemPackA<Problem>() == 16 &&
               GetSmemPackB<Problem>() == 16;
    }();

    // Assume normal LDS layout, not transpose-load
    template <typename Problem>
    static constexpr bool UseXorSwizzle =
        !is_a_load_tr<Problem> && !is_b_load_tr<Problem> &&
        IsSupportedXorSwizzleDataType<Problem> && IsSupportedXorSwizzleAsyncWidth<Problem> &&
        IsSupportedXorSwizzleShape<Problem>;

    template <typename Problem, index_t MNPerBlock, index_t K2>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXorSwizzleABDramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t KWarps    = BlockWarps::at(I2);
        constexpr index_t K1        = WarpTile::at(I2) / K2;
        constexpr index_t K0        = KPerBlock / (KWarps * K1 * K2);

        constexpr index_t warp_size = get_warp_size();
        constexpr index_t warp_num  = BlockSize / warp_size;

        static_assert(KWarps == 1, "MX XOR swizzle currently supports KWarps == 1");
        static_assert(KWarps * K0 * K1 * K2 == KPerBlock, "Wrong!");

        constexpr index_t M2 = warp_size / K1;
        constexpr index_t M1 = warp_num / Problem::NumWaveGroups;
        constexpr index_t M0 = MNPerBlock / (M1 * M2);

        static_assert(M0 * M1 * M2 == MNPerBlock, "Wrong!");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t KPack     = GetXorSwizzleKPackA<Problem>();
            return MakeXorSwizzleABDramTileDistribution<Problem, MPerBlock, KPack>();
        }
        else
        {
            return UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>::
                template MakeADramTileDistribution<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPack     = GetXorSwizzleKPackB<Problem>();
            return MakeXorSwizzleABDramTileDistribution<Problem, NPerBlock, KPack>();
        }
        else
        {
            return UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>::
                template MakeBDramTileDistribution<Problem>();
        }
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
        static_assert(KWarps == 1, "MX XOR swizzle currently supports KWarps == 1");
        static_assert(wg_attr_num_access_v == 1 || wg_attr_num_access_v == 2,
                      "MX XOR swizzle currently supports FP8, BF8, FP4");

        constexpr index_t K2Pad = K2 < 16 ? 16 : K2;
        constexpr index_t M3    = 4;
        constexpr index_t M2    = warp_size / K1 / M3;
        constexpr index_t M1    = WarpTileMN / (M2 * M3);
        constexpr index_t M0    = MNPerBlock / (M1 * M2 * M3);

        static_assert(M0 * M1 * M2 * M3 == MNPerBlock, "Wrong!");

        constexpr index_t PadSize = 4 * K2;

        constexpr auto desc_0 = make_naive_tensor_descriptor(
            number_tuple<M0, K0, M1, M2, M3, K1, K2>{},
            number_tuple<K0*(M1 * (M2 * M3 * K1 * K2Pad) + (M1 - 1) * PadSize),
                         M1*(M2 * M3 * K1 * K2Pad) + (M1 - 1) * PadSize,
                         M2 * M3 * K1 * K2Pad + PadSize,
                         M3 * K1 * K2Pad,
                         K1 * K2Pad,
                         K2Pad,
                         1>{},
            number<K2>{},
            number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M2>{}),
                       make_xor_transform(make_tuple(number<M3>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}));

        constexpr auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M3>{}),
                       make_merge_transform_v3_division_mod(number_tuple<K0, K1, K2>{})),
            make_tuple(sequence<0, 2, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_2;
    }

    // MX scaling configuration: each e8m0 scale covers 32 elements in K
    static constexpr int BlockScaleSize = 32;

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
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
                constexpr index_t KPack = GetXorSwizzleKPackA<Problem>();
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
                constexpr index_t KPack = GetSmemPackA<Problem>();

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
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
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
                constexpr index_t KPack = GetXorSwizzleKPackB<Problem>();
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
                constexpr index_t KPack = GetSmemPackB<Problem>();

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
    }

    // MX GEMM: Double access for FP8/BF8, Single for FP4
    template <typename DataType_>
    CK_TILE_HOST_DEVICE static constexpr auto CalculateWGAttrNumAccess()
    {
        using DataType = remove_cvref_t<DataType_>;

        if constexpr(std::is_same_v<DataType, fp8_t> || std::is_same_v<DataType, bf8_t>)
        {
            return WGAttrNumAccessEnum::Double;
        }
        else if constexpr(std::is_same_v<DataType, pk_fp4_t>)
        {
            return WGAttrNumAccessEnum::Single;
        }
        else
        {
            static_assert(sizeof(DataType) == 0,
                          "CalculateWGAttrNumAccess(): unsupported data type");
            return WGAttrNumAccessEnum::Invalid;
        }
    }

    // Get number of accesses
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWGAttrNumAccess()
    {
        constexpr auto num_access_a = CalculateWGAttrNumAccess<typename Problem::ADataType>();
        constexpr auto num_access_b = CalculateWGAttrNumAccess<typename Problem::BDataType>();

        if constexpr(static_cast<index_t>(num_access_a) >= static_cast<index_t>(num_access_b))
        {
            return num_access_a;
        }
        else
        {
            return num_access_b;
        }
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

        constexpr index_t M4 = 4; // same as MakeXorSwizzledABLdsBlockDescriptor::M3
        static_assert(get_warp_size() % (wg_attr_num_access_v * K1 * M4) == 0,
                      "warp_size must be divisible by (wg_attr_num_access_v * K1 * M4)");

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
            constexpr index_t KPack = GetXorSwizzleKPackA<Problem>();
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
            constexpr index_t KPack = GetXorSwizzleKPackB<Problem>();
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
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        using ADataType = typename Problem::ADataType;
        using BDataType = typename Problem::BDataType;
        using CDataType = typename Problem::CDataType;

        // FP4 and FP8 require different layouts for the scaled mfma instructions
        constexpr auto wg_attr_num_access = GetWGAttrNumAccess<Problem>();

        using WarpGemm = WarpGemmDispatcher<ADataType,
                                            BDataType,
                                            CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockMXGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }

    // XdlPack: how many e8m0_t scale values are packed into one int32_t per dimension
    // Host packs MXdlPack * KXdlPack e8m0_t into one int32_t for A scales
    // Host packs NXdlPack * KXdlPack e8m0_t into one int32_t for B scales
    static constexpr int MXdlPack = 2;
    static constexpr int NXdlPack = 2;
    static constexpr int KXdlPack = 2;

    // MX Scale tile distributions for loading pre-packed int32_t from global memory
    // Packed layout: [M/MXdlPack, K/32/KXdlPack] of int32_t
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t MWarp     = BlockWarps::at(number<0>{});
        constexpr index_t NWarp     = BlockWarps::at(number<1>{});
        constexpr index_t MPerXdl   = WarpTile::at(number<0>{});
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K_Lane       = get_warp_size() / MPerXdl;
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * MPerXdl);
        constexpr index_t KPerXdl      = WarpTile::at(number<2>{});
        constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
        constexpr index_t KPerLane     = KPerXdl / BlockScaleSize / K_Lane;

        // Effective pack sizes: fall back to 1 when iteration count < pack size
        constexpr index_t MXdlPackEff =
            (MIterPerWarp >= MXdlPack && MIterPerWarp % MXdlPack == 0) ? MXdlPack : 1;
        constexpr index_t KXdlPackEff =
            (KIterPerWarp >= KXdlPack && KIterPerWarp % KXdlPack == 0) ? KXdlPack : 1;

        constexpr index_t MIterPerWarp_packed = MIterPerWarp / MXdlPackEff;
        constexpr index_t KIterPerWarp_packed = KIterPerWarp / KXdlPackEff;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MWarp, MIterPerWarp_packed, MPerXdl>,
                                             sequence<KIterPerWarp_packed, K_Lane, KPerLane>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 0>, sequence<1, 2>>,
                                       sequence<2, 1, 2>,
                                       sequence<0, 1, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t NPerBlock    = Problem::BlockGemmShape::kN;
        constexpr index_t MWarp        = BlockWarps::at(number<0>{});
        constexpr index_t NWarp        = BlockWarps::at(number<1>{});
        constexpr index_t NPerXdl      = WarpTile::at(number<1>{});
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t K_Lane       = get_warp_size() / NPerXdl;
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * NPerXdl);

        constexpr index_t KPerXdl      = WarpTile::at(number<2>{});
        constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
        constexpr index_t KPerLane     = KPerXdl / BlockScaleSize / K_Lane;

        // Effective pack sizes: fall back to 1 when iteration count < pack size
        constexpr index_t NXdlPackEff =
            (NIterPerWarp >= NXdlPack && NIterPerWarp % NXdlPack == 0) ? NXdlPack : 1;
        constexpr index_t KXdlPackEff =
            (KIterPerWarp >= KXdlPack && KIterPerWarp % KXdlPack == 0) ? KXdlPack : 1;

        constexpr index_t NIterPerWarp_packed = NIterPerWarp / NXdlPackEff;
        constexpr index_t KIterPerWarp_packed = KIterPerWarp / KXdlPackEff;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NWarp, NIterPerWarp_packed, NPerXdl>,
                                             sequence<KIterPerWarp_packed, K_Lane, KPerLane>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 0>, sequence<1, 2>>,
                                       sequence<2, 1, 2>,
                                       sequence<0, 1, 2>>{});
    }
};
} // namespace ck_tile
