// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
struct GemmPipelineAgBgCrDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrDefaultPolicy>
{
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrDefaultPolicy>;

    template <typename Problem>
    using LdsADataType = typename Problem::ADataType;

    template <typename Problem>
    using LdsBDataType = typename Problem::BDataType;

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t MaxVecSize         = get_max_mem_vec_inst_width();

    // The swizzle factor is defined base on the number of contiguous lanes
    // in the instruction.
    template <typename DataType, typename WarpTile, bool IsTranspose, index_t KPack>
    CK_TILE_DEVICE static constexpr auto get_swizzle_factor(number<KPack>)
    {
        constexpr index_t WarpTileK = WarpTile::at(I2);
        constexpr bool is_8bit_float =
            std::is_same_v<DataType, fp8_t> || std::is_same_v<DataType, bf8_t>;
        constexpr index_t NumAccess = is_8bit_float ? 2 : 1;

        return IsTranspose ? 2 * sizeof(DataType) : WarpTileK / KPack / NumAccess;
    };

    template <index_t YPerBlock,
              typename DataType,
              typename WarpTile,
              typename WindowTmp,
              bool IsTranspose>
    CK_TILE_DEVICE static constexpr auto MakeDramTensorView(const WindowTmp& window_tmp,
                                                            bool_constant<IsTranspose>)
    {
        constexpr index_t PackedSize = numeric_traits<DataType>::PackedSize;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view  = window_tmp.get_bottom_tensor_view();
        const auto [Xs, Ys] = tensor_view.get_tensor_descriptor().get_lengths();

        constexpr index_t ContiguousElementsCacheLine =
            (kDramLoadPackBytes / sizeof(DataType) * PackedSize);
        constexpr index_t NumContiguousElements = std::min(ContiguousElementsCacheLine, YPerBlock);
        constexpr index_t KPack                 = MaxVecSize / sizeof(DataType) * PackedSize;
        constexpr index_t PacksPerLdsRow        = NumContiguousElements / KPack;

        constexpr index_t Y2 = KPack;
        constexpr index_t Y1 = PacksPerLdsRow;
        const index_t Y0     = integer_divide_ceil(Ys, Y1 * Y2);
        const auto Y_lens    = make_tuple(Y0, number<Y1>{}, number<Y2>{});

        constexpr index_t X1 = get_swizzle_factor<DataType, WarpTile, IsTranspose>(number<KPack>{});
        const index_t X0     = integer_divide_ceil(Xs, X1);
        const auto X_lens    = make_tuple(X0, number<X1>{});

        const auto& desc_0 = tensor_view.get_tensor_descriptor();

        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_unmerge_transform(X_lens), make_unmerge_transform(Y_lens)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}));

        const auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_pass_through_transform(X0),
                       make_xor_transform(make_tuple(number<X1>{}, number<Y1>{})),
                       make_pass_through_transform(Y0),
                       make_pass_through_transform(number<Y2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));

        const auto desc =
            transform_tensor_descriptor(desc_2,
                                        make_tuple(make_merge_transform_v3_division_mod(X_lens),
                                                   make_merge_transform_v3_division_mod(Y_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr = &(tensor_view.get_buffer_view()(0));
        return make_tensor_view<address_space_enum::global>(byte_ptr, desc);
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeADramTensorView(const WindowTmp& window_tmp)
    {
        if constexpr(Problem::Async)
        {
            using ADataType             = remove_cvref_t<typename Problem::ADataType>;
            using WarpTile              = typename Problem::BlockGemmShape::WarpTile;
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t YPerBlock = is_a_load_tr<Problem> ? KPerBlock : MPerBlock;
            using IsTranspose           = bool_constant<is_a_load_tr<Problem>>;

            return MakeDramTensorView<YPerBlock, ADataType, WarpTile>(window_tmp, IsTranspose{});
        }
        else
        {
            return window_tmp.get_bottom_tensor_view();
        }
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeBDramTensorView(const WindowTmp& window_tmp)
    {
        if constexpr(Problem::Async)
        {
            using BDataType             = remove_cvref_t<typename Problem::BDataType>;
            using WarpTile              = typename Problem::BlockGemmShape::WarpTile;
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t YPerBlock = is_b_load_tr<Problem> ? KPerBlock : NPerBlock;
            using IsTranspose           = bool_constant<is_b_load_tr<Problem>>;

            return MakeDramTensorView<YPerBlock, BDataType, WarpTile>(window_tmp, IsTranspose{});
        }
        else
        {
            return window_tmp.get_bottom_tensor_view();
        }
    }

    template <typename DataType,
              index_t MNPerBlock,
              index_t KPerBlock,
              index_t BlockSize,
              bool IsTranspose>
    CK_TILE_DEVICE static constexpr auto MakeDramTileDistribution(bool_constant<IsTranspose>)
    {
        constexpr index_t WaveSize   = get_warp_size();
        constexpr index_t PackedSize = numeric_traits<DataType>::PackedSize;
        constexpr index_t XPerBlock  = IsTranspose ? KPerBlock : MNPerBlock;
        constexpr index_t YPerBlock  = IsTranspose ? MNPerBlock : KPerBlock;

        constexpr index_t ContiguousElementsCacheLine =
            (kDramLoadPackBytes / sizeof(DataType) * PackedSize);
        constexpr index_t NumContiguousElements = std::min(ContiguousElementsCacheLine, YPerBlock);
        constexpr index_t KPack                 = MaxVecSize / sizeof(DataType) * PackedSize;
        constexpr index_t PacksPerLdsRow        = NumContiguousElements / KPack;

        constexpr index_t Y2 = KPack;
        constexpr index_t Y1 = PacksPerLdsRow;
        constexpr index_t Y0 = YPerBlock / (Y2 * Y1);
        constexpr index_t X2 = WaveSize / Y1;
        constexpr index_t X1 = BlockSize / WaveSize;
        constexpr index_t X0 = XPerBlock / (X1 * X2);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<X0, X1, X2>, sequence<Y0, Y1, Y2>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        if constexpr(Problem::Async)
        {
            using ADataType             = remove_cvref_t<typename Problem::ADataType>;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t BlockSize = Problem::kBlockSize;
            using IsTranspose           = bool_constant<is_a_load_tr<Problem>>;

            return MakeDramTileDistribution<ADataType, MPerBlock, KPerBlock, BlockSize>(
                IsTranspose{});
        }
        else
        {
            return Base::template MakeADramTileDistribution<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        if constexpr(Problem::Async)
        {
            using BDataType             = remove_cvref_t<typename Problem::BDataType>;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t BlockSize = Problem::kBlockSize;
            using IsTranspose           = bool_constant<is_b_load_tr<Problem>>;

            return MakeDramTileDistribution<BDataType, NPerBlock, KPerBlock, BlockSize>(
                IsTranspose{});
        }
        else
        {
            return Base::template MakeBDramTileDistribution<Problem>();
        }
    }

    template <index_t MNPerBlock,
              index_t KPerBlock,
              typename DataType,
              typename WarpTile,
              bool IsTranspose>
    CK_TILE_DEVICE static constexpr auto MakeLdsBlockDescriptor(bool_constant<IsTranspose>)
    {
        constexpr index_t XPerBlock = IsTranspose ? KPerBlock : MNPerBlock;
        constexpr index_t YPerBlock = IsTranspose ? MNPerBlock : KPerBlock;
        // MPerXdl == NPerXdl always
        constexpr index_t XPerXdl = IsTranspose ? WarpTile::at(I2) : WarpTile::at(I0);

        constexpr index_t PackedSize = numeric_traits<DataType>::PackedSize;
        constexpr index_t ContiguousElementsCacheLine =
            (kDramLoadPackBytes / sizeof(DataType) * PackedSize);
        constexpr index_t NumContiguousElements = std::min(ContiguousElementsCacheLine, YPerBlock);
        constexpr index_t KPack                 = MaxVecSize / sizeof(DataType) * PackedSize;
        constexpr index_t PacksPerLdsRow        = NumContiguousElements / KPack;

        constexpr index_t Y2 = KPack;
        constexpr index_t Y1 = PacksPerLdsRow;
        constexpr index_t Y0 = YPerBlock / (Y1 * Y2);
        static_assert(Y0 * Y1 * Y2 == YPerBlock, "Y0, Y1, Y2 must cover whole YPerBlock!");

        constexpr index_t WaveSize = get_warp_size();

        constexpr index_t X3 = get_swizzle_factor<DataType, WarpTile, IsTranspose>(number<KPack>{});
        constexpr index_t X2 = WaveSize / Y1 / X3;
        constexpr index_t X1 = XPerXdl / (X2 * X3);
        constexpr index_t X0 = XPerBlock / (X1 * X2 * X3);
        static_assert(X0 * X1 * X2 * X3 == XPerBlock, "X0, X1, X2, X3 must cover whole XPerBlock!");

        // We pad the tensor layout to fix bank conflicts only within
        // NumContiguousElements x MNPerXdl tile to minimize padding needed.
        // X0 and Y0 define the number of such tiles in both dimensions, so the stride for those
        // dimensions take into account only the padding within the above defined tile

        constexpr index_t Pad = X3 * Y2;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<X0>{},
                       number<Y0>{},
                       number<X1>{},
                       number<X2>{},
                       number<X3>{},
                       number<Y1>{},
                       number<Y2>{}),
            make_tuple(number<Y0*(X1 * (X2 * X3 * Y1 * Y2) + (X1 - 1) * Pad)>{},
                       number<X1*(X2 * X3 * Y1 * Y2) + (X1 - 1) * Pad>{},
                       number<X2 * X3 * Y1 * Y2 + Pad>{},
                       number<X3 * Y1 * Y2>{},
                       number<Y1 * Y2>{},
                       number<Y2>{},
                       number<1>{}),
            number<Y2>{},
            number<1>{});

        constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<X0>{}),
                       make_pass_through_transform(number<Y0>{}),
                       make_pass_through_transform(number<X1>{}),
                       make_pass_through_transform(number<X2>{}),
                       make_xor_transform(make_tuple(number<X3>{}, number<Y1>{})),
                       make_pass_through_transform(number<Y2>{})),
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

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<X0>{}, number<X1>{}, number<X2>{}, number<X3>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<Y0>{}, number<Y1>{}, number<Y2>{}))),
            make_tuple(sequence<0, 2, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        if constexpr(Problem::Async)
        {
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            using WarpTile              = typename Problem::BlockGemmShape::WarpTile;
            using IsTranspose           = bool_constant<is_a_load_tr<Problem>>;

            return MakeLdsBlockDescriptor<MPerBlock, KPerBlock, OverrideADataType, WarpTile>(
                IsTranspose{});
        }
        else
        {
            return Base::template MakeALdsBlockDescriptor<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        if constexpr(Problem::Async)
        {
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            using WarpTile              = typename Problem::BlockGemmShape::WarpTile;
            using BDataType             = remove_cvref_t<typename Problem::BDataType>;
            using IsTranspose           = bool_constant<is_b_load_tr<Problem>>;

            return MakeLdsBlockDescriptor<NPerBlock, KPerBlock, BDataType, WarpTile>(IsTranspose{});
        }
        else
        {
            return Base::template MakeBLdsBlockDescriptor<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

#if defined(__gfx950__)
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::AComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access_A =
            !(is_a_load_tr<Problem>)             ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
        constexpr auto wg_attr_num_access_B =
            !(is_b_load_tr<Problem>)             ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
#else
        constexpr auto wg_attr_num_access_A = WGAttrNumAccessEnum::Default;
        constexpr auto wg_attr_num_access_B = WGAttrNumAccessEnum::Default;
#endif

        using ATypeToUse = typename Problem::AComputeDataType;
        using BTypeToUse = typename Problem::BComputeDataType;

        using WarpGemm = WarpGemmDispatcher<ATypeToUse,
                                            BTypeToUse,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            Problem::UseStructuredSparsity,
                                            wg_attr_num_access_A,
                                            wg_attr_num_access_B>;

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ATypeToUse,
                                                                      BTypeToUse,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;

        using BlockGemm = BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>;
        return BlockGemm{};
    }
};

} // namespace ck_tile
