// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_eight_waves_v1.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsyncEightWaves
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
namespace detail {
template <typename Problem>
struct GemmPipelineAgBgCrCompAsyncEightWavesPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t kDramLoadPackBytes = 128;

    using ALayout          = remove_cvref_t<typename Problem::ALayout>;
    using BLayout          = remove_cvref_t<typename Problem::BLayout>;
    using ADataType        = remove_cvref_t<typename Problem::ADataType>;
    using BDataType        = remove_cvref_t<typename Problem::BDataType>;
    using CDataType        = remove_cvref_t<typename Problem::CDataType>;
    using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
    using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
    using ComputeDataType  = AComputeDataType;
    static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>,
                  "ALayout must be RowMajor!");
    static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>,
                  "BLayout must be ColumnMajor!");
    static_assert(
        is_any_of<AComputeDataType, fp8_t, bf8_t, pk_fp4_t, pk_fp6x16_t, fp16_t, bf16_t, int8_t>::
            value);
    static_assert(
        is_any_of<BComputeDataType, fp8_t, bf8_t, pk_fp4_t, pk_fp6x16_t, fp16_t, bf16_t, int8_t>::
            value);
    static_assert(std::is_same_v<AComputeDataType, BComputeDataType>);

    static constexpr auto WGAccess =
        std::is_same_v<ComputeDataType, fp8_t> || std::is_same_v<ComputeDataType, bf8_t>
            ? WGAttrNumAccessEnum::Double
            : WGAttrNumAccessEnum::Single;
    static constexpr auto PackedSize = numeric_traits<ComputeDataType>::PackedSize;

    using BlockGemmShape = typename Problem::BlockGemmShape;
    using BlockWarps     = typename BlockGemmShape::BlockWarps;
    using WarpTile       = typename BlockGemmShape::WarpTile;

    // Check if Preshuffle or PreshuffleB exists. In this way it will work for both GEMM and ABQuant
    template <typename T>
    using has_preshuffle_type = decltype(T::Preshuffle);
    template <typename T>
    using has_preshuffleb_type = decltype(T::PreshuffleB);

    static constexpr bool IsPreshuffle_ = [] {
        if constexpr(is_detected<has_preshuffle_type, Problem>{})
        {
            return Problem::Preshuffle;
        }
        else
        {
            return false;
        }
    }();

    static constexpr bool IsPreshuffleB_ = [] {
        if constexpr(is_detected<has_preshuffleb_type, Problem>{})
        {
            return Problem::PreshuffleB;
        }
        else
        {
            return false;
        }
    }();

    static constexpr bool Preshuffle    = IsPreshuffle_ || IsPreshuffleB_;
    static constexpr index_t BlockSize  = Problem::kBlockSize;
    static constexpr index_t MPerBlock  = BlockGemmShape::kM;
    static constexpr index_t NPerBlock  = BlockGemmShape::kN;
    static constexpr index_t KPerBlock  = BlockGemmShape::kK;
    static constexpr index_t WarpTileM  = WarpTile::at(I0);
    static constexpr index_t WarpTileN  = WarpTile::at(I1);
    static constexpr index_t WarpTileK  = WarpTile::at(I2);
    static constexpr index_t MWarpTiles = MPerBlock / WarpTileM;
    static constexpr index_t NWarpTiles = NPerBlock / WarpTileN;
    static constexpr index_t KWarpTiles = KPerBlock / WarpTileK;

    static constexpr index_t MWarps       = BlockWarps::at(I0);
    static constexpr index_t NWarps       = BlockWarps::at(I1);
    static constexpr index_t KWarps       = BlockWarps::at(I2);
    static constexpr index_t MIterPerWarp = MWarpTiles / MWarps;
    static constexpr index_t NIterPerWarp = NWarpTiles / NWarps;
    static constexpr index_t NPerWarp     = NPerBlock / NWarps;
    static_assert(NWarps == 2, "NWarps == 2 for ping-pong!");

    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t warp_num  = BlockSize / warp_size;
    static_assert(warp_size == 64, "Wrong!");
    static_assert(warp_num * warp_size == BlockSize, "Wrong!");

    static_assert(sizeof(ADataType) == sizeof(BDataType), "Wrong!");
    static_assert(sizeof(ADataType) == sizeof(ComputeDataType));
    static constexpr index_t ElementSize = sizeof(ComputeDataType);
    static constexpr index_t K2          = Problem::VectorLoadSize / ElementSize * PackedSize;
    // We define kDramLoadPackElems as 128 for fp6 because K2 == 16, so in this way we have correct
    // values for K1 (number of contiguous lanes in a row)
    static constexpr index_t kDramLoadPackElems =
        std::is_same_v<ComputeDataType, pk_fp6x16_t>
            ? kDramLoadPackBytes
            : kDramLoadPackBytes / sizeof(ComputeDataType) * PackedSize;
    static constexpr index_t PacksPerLdsRow = std::min(kDramLoadPackElems, KPerBlock) / K2;
    static constexpr index_t K1             = PacksPerLdsRow;
    static constexpr index_t K0             = KPerBlock / (K1 * K2);
    static_assert(K0 * K1 * K2 == KPerBlock, "Wrong!");

    static constexpr index_t SwizzleFactor = WarpTileK / static_cast<index_t>(WGAccess) / K2;

    CK_TILE_DEVICE static constexpr bool IsPreshuffle() { return Preshuffle; }

    CK_TILE_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t M2 = warp_size / K1;
        constexpr index_t M1 = warp_num;
        constexpr index_t M0 = MPerBlock / M1 / M2;
        static_assert(M0 * M1 * M2 == MPerBlock, "wrong!");

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<M0, M1, M2>, ck_tile::sequence<K0, K1, K2>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>, // M0 M2,K1
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2, 1>>,
                ck_tile::sequence<1, 2, 2>, // M0,K0,K2
                ck_tile::sequence<0, 0, 2>>{});
    }

    CK_TILE_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        if constexpr(Preshuffle)
        {
            constexpr index_t K1_ = warp_size;
            constexpr index_t K0_ = KPerBlock * WarpTileN / K1_ / K2;
            static_assert(K0_ * K1_ * K2 == KPerBlock * WarpTileN, "wrong!");

            constexpr index_t N1 = warp_num / NWarps / K0_;
            constexpr index_t N0 = NPerBlock / WarpTileN / N1 / NWarps;
            static_assert(NWarps * N0 * N1 == NPerBlock / WarpTileN, "wrong!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<NWarps, N0, N1>, sequence<K0_, K1_, K2>>,
                                           tuple<sequence<1, 1, 2>, sequence<2>>, // NWarps,N1,K0 K1
                                           tuple<sequence<0, 2, 0>, sequence<1>>,
                                           sequence<1, 2>, // N0,K2
                                           sequence<1, 2>>{});
        }
        else
        {
            constexpr index_t N2 = warp_size / K1;
            constexpr index_t N1 = warp_num / NWarps;
            constexpr index_t N0 = NPerBlock / N1 / N2 / NWarps;
            static_assert(NWarps * N0 * N1 * N2 == NPerBlock, "wrong!");

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<NWarps, N0, N1, N2>, sequence<K0, K1, K2>>,
                    tuple<sequence<1, 1>, sequence<1, 2>>, // NWarps,N1 N2,K1
                    tuple<sequence<0, 2>, sequence<3, 1>>,
                    sequence<1, 2, 2>, // N0,K0,K2
                    sequence<1, 0, 2>>{});
        }
    }

    template <typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        const index_t k_tiles = cols / (KWarps * K1 * K2);
        const auto col_lens   = make_tuple(k_tiles, number<KWarps>{}, number<K1>{}, number<K2>{});

        constexpr index_t M1 = SwizzleFactor;
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        // Build the 6D view by composing unmerge transforms on top of the
        // input view's existing descriptor. This preserves the input's actual
        // strides (so a non-packed leading-dim stride is honored) and inherits
        // its element_space_size for bounds checking.
        const auto desc_0 = transform_tensor_descriptor(
            tensor_view_tmp.get_tensor_descriptor(),
            make_tuple(make_unmerge_transform(row_lens), make_unmerge_transform(col_lens)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}));
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(k_tiles),
                       make_pass_through_transform(number<KWarps>{}),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}));
        const auto desc = transform_tensor_descriptor( //
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(row_lens),
                       make_merge_transform_v3_division_mod(col_lens)),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tile_window(make_tensor_view<address_space_enum::global>(
                                    &tensor_view_tmp.get_buffer_view()(0), desc),
                                window_tmp.get_window_lengths(),
                                window_tmp.get_window_origin());
    }

    template <typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadBDramWindow(const WindowTmp& window_tmp)
    {
        if constexpr(!Preshuffle)
            return MakeAsyncLoadADramWindow(window_tmp);
        else
            return make_tile_window(window_tmp.get_bottom_tensor_view(),
                                    number_tuple<NPerBlock / WarpTileN, KPerBlock * WarpTileN>{},
                                    window_tmp.get_window_origin());
    }

    template <index_t MNPerBlock, index_t warp_groups_>
    CK_TILE_DEVICE static constexpr auto MakeABLdsBlockDescriptor_()
    {
        constexpr index_t M4 = SwizzleFactor;
        constexpr index_t M3 = warp_size / (M4 * K1);
        constexpr index_t M2 = WarpTileM / M4 / M3;
        constexpr index_t M1 = (warp_num / warp_groups_) / M2;
        constexpr index_t M0 = MNPerBlock / M1 / M2 / M3 / M4;

        static_assert(M1 * M0 * M2 * M3 * M4 == MNPerBlock, "wrong!");

        constexpr index_t PadSize = SwizzleFactor * K2;

        // Padding is needed between waves writing to LDS for a single mfma tile
        // Example: instruction 16x128 8 bit
        // 2 waves read elements from gmem to lds but they are consumed by a single
        // wave when reading from LDS, so we need padding there but not between
        // consecutive mfma tiles in LDS
        constexpr auto desc_0 = make_naive_tensor_descriptor( //
            make_tuple(number<M0>{},
                       number<K0>{},
                       number<M1>{},
                       number<M2>{},
                       number<M3>{},
                       number<M4>{},
                       number<K1>{},
                       number<K2>{}),
            make_tuple(number<K0 * M1*(M2 * (M3 * M4 * K1 * K2) + (M2 - 1) * PadSize)>{},
                       number<M1*(M2 * (M3 * M4 * K1 * K2) + (M2 - 1) * PadSize)>{},
                       number<M2*(M3 * M4 * K1 * K2) + (M2 - 1) * PadSize>{},
                       number<M3 * M4 * K1 * K2 + PadSize>{},
                       number<M4 * K1 * K2>{},
                       number<K1 * K2>{},
                       number<K2>{},
                       number<1>{}),
            number<K2>{},
            number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M2>{}),
                       make_pass_through_transform(number<M3>{}),
                       make_xor_transform(make_tuple(number<M4>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            container_concat(generate_tuple([](auto i) { return sequence<i>{}; }, number<5>{}),
                             make_tuple(sequence<5, 6>{}),
                             make_tuple(sequence<7>{})),
            container_concat(generate_tuple([](auto i) { return sequence<i>{}; }, number<5>{}),
                             make_tuple(sequence<5, 6>{}),
                             make_tuple(sequence<7>{})));

        constexpr auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M3, M4>{}),
                       make_merge_transform_v3_division_mod(number_tuple<K0, K1, K2>{})),
            make_tuple(sequence<0, 2, 3, 4, 5>{}, sequence<1, 6, 7>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return desc_2;
    }
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return MakeABLdsBlockDescriptor_<MPerBlock, 1>();
    }
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        if constexpr(!Preshuffle)
            return MakeABLdsBlockDescriptor_<NPerBlock, 2>();
        else
        {
            constexpr index_t K1_ = warp_size;                        // 64
            constexpr index_t K0_ = KPerBlock * WarpTileN / K1_ / K2; // 2
            static_assert(K0_ * K1_ * K2 == KPerBlock * WarpTileN, "wrong!");

            constexpr index_t N1 = warp_num / NWarps / K0_;             // 2
            constexpr index_t N0 = NPerBlock / WarpTileN / N1 / NWarps; // 4
            static_assert(NWarps * N0 * N1 == NPerBlock / WarpTileN, "wrong!");

            constexpr auto desc_0 =
                make_naive_tensor_descriptor_packed(number_tuple<NWarps, N1, K0_, N0, K1_, K2>{});
            constexpr auto desc_1 = transform_tensor_descriptor(
                desc_0,
                make_tuple(make_merge_transform_v3_division_mod(number_tuple<NWarps, N0, N1>{}),
                           make_merge_transform_v3_division_mod(number_tuple<K0_, K1_, K2>{})),
                make_tuple(sequence<0, 3, 1>{}, sequence<2, 4, 5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return desc_1;
        }
    }
    CK_TILE_DEVICE static constexpr auto MakeBLdsReadBlockDescriptor()
    {
        if constexpr(!Preshuffle)
            return MakeABLdsBlockDescriptor_<NPerBlock, 2>();
        else
        {
            constexpr index_t K1_ = warp_size / WarpTileN; // 4
            constexpr index_t K0_ = KPerBlock / K1_ / K2;  // 2
            static_assert(K0_ * K1_ * K2 == KPerBlock, "wrong!");

            constexpr index_t N2 = warp_size / K1_;              // 16
            constexpr index_t N1 = warp_num / NWarps / K0_;      // 2
            constexpr index_t N0 = NPerBlock / N1 / N2 / NWarps; // 4
            static_assert(NWarps * N0 * N1 * N2 == NPerBlock, "wrong!");

            constexpr auto desc_0 = make_naive_tensor_descriptor_packed(
                number_tuple<NWarps, N1, K0_, N0, K1_, N2, K2>{});
            constexpr auto desc_1 = transform_tensor_descriptor(
                desc_0,
                make_tuple(make_merge_transform_v3_division_mod(number_tuple<NWarps, N0, N1, N2>{}),
                           make_merge_transform_v3_division_mod(number_tuple<K0_, K1_, K2>{})),
                make_tuple(sequence<0, 3, 1, 5>{}, sequence<2, 4, 6>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return desc_1;
        }
    }
    static_assert(MakeBLdsBlockDescriptor().get_element_space_size() ==
                      MakeBLdsReadBlockDescriptor().get_element_space_size(),
                  "Wrong!");

    CK_TILE_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t sizeofType =
            std::is_same_v<AComputeDataType, pk_fp6x16_t> ? 16 : sizeof(AComputeDataType);
        constexpr index_t desc_size = MakeALdsBlockDescriptor().get_element_space_size();
        return integer_least_multiple(sizeofType * desc_size / PackedSize, 16);
    }
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t sizeofType =
            std::is_same_v<BComputeDataType, pk_fp6x16_t> ? 16 : sizeof(BComputeDataType);
        constexpr index_t desc_size = MakeBLdsBlockDescriptor().get_element_space_size();
        return integer_least_multiple(sizeofType * desc_size / PackedSize, 16);
    }

    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * (GetSmemSizeA() + GetSmemSizeB());
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeA() { return K2; }
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeB() { return K2; }
    CK_TILE_DEVICE static constexpr auto GetSmemPackA() { return K2; }
    CK_TILE_DEVICE static constexpr auto GetSmemPackB() { return K2; }

    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // TODO: Fix for transpose
        constexpr auto wg_attr_num_access = WGAccess;

        using WarpGemm = WarpGemmDispatcher<typename Problem::AComputeDataType,
                                            typename Problem::BComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::AComputeDataType,
                                                typename Problem::BComputeDataType,
                                                typename Problem::CDataType,
                                                BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegEightWavesV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace detail

struct GemmPipelineAgBgCrCompAsyncEightWavesPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncEightWavesPolicy>
{
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncEightWavesPolicy>;
    using Base::is_a_load_tr;
    using Base::is_b_load_tr;
#define FORWARD_METHOD_(method)                                                      \
    template <typename Problem, typename... Args>                                    \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)                 \
    {                                                                                \
        return detail::GemmPipelineAgBgCrCompAsyncEightWavesPolicy<Problem>::method( \
            std::forward<Args>(args)...);                                            \
    }

    FORWARD_METHOD_(GetBlockGemm);
    FORWARD_METHOD_(MakeADramTileDistribution);
    FORWARD_METHOD_(MakeBDramTileDistribution);
    FORWARD_METHOD_(MakeAsyncLoadADramWindow);
    FORWARD_METHOD_(MakeAsyncLoadBDramWindow);
    FORWARD_METHOD_(MakeALdsBlockDescriptor);
    FORWARD_METHOD_(MakeBLdsBlockDescriptor);
    FORWARD_METHOD_(MakeBLdsReadBlockDescriptor);
    FORWARD_METHOD_(GetSmemSizeA);
    FORWARD_METHOD_(GetSmemSizeB);
    FORWARD_METHOD_(GetSmemSize);
    FORWARD_METHOD_(GetVectorSizeA);
    FORWARD_METHOD_(GetVectorSizeB);
    FORWARD_METHOD_(GetSmemPackA);
    FORWARD_METHOD_(GetSmemPackB);
    FORWARD_METHOD_(IsPreshuffle);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
