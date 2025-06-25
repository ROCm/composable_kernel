// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
namespace ck_tile {

// TODO: need to check lds bank conflict
template <bool TransLoadEn_, bool AsyncCopy_, bool AsyncStore_, index_t Stages_>
struct GemmPipelineWmmaPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    // use this variable to decide whether use transpose load(such as DS_LOAD_TR16_B128) or not; the
    // below is used in gfx12/gfx13; when A is Column Major, will still use the similar layout as
    // global memory; the same as B in Row Major.
    static constexpr bool TransLoadEn = TransLoadEn_;
    // the below is used in gfx13 to use async copy from global to lds and async store from lds to
    // global
    static constexpr bool AsyncCopy  = AsyncCopy_;
    static constexpr bool AsyncStore = AsyncStore_; // AsyncStore is used in Epilogue

    static constexpr index_t NumLdsNumA = Stages_;
    static constexpr index_t NumLdsNumB = Stages_;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleASmemElementSpace()
    {
        constexpr index_t kMPerBlock  = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock  = Problem::BlockGemmShape::kK;
        constexpr index_t kMKPerBlock = kMPerBlock * kKPerBlock;
        return kMKPerBlock;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleBSmemElementSpace()
    {
        constexpr index_t kNPerBlock  = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock  = Problem::BlockGemmShape::kK;
        constexpr index_t kNKPerBlock = kNPerBlock * kKPerBlock;
        return kNKPerBlock;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t kMKPerBlock = GetSingleASmemElementSpace<Problem>();

        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        constexpr bool is_a_column_major =
            std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>;

        constexpr auto alignment = 8; // TODO : need to change for different data type
        if constexpr(is_a_column_major)
        {
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumLdsNumA>{},
                           number<kKPerBlock>{},
                           number<kMPerBlock / alignment>{},
                           number<alignment>{}),
                make_tuple(
                    number<kMKPerBlock>{}, number<kMPerBlock>{}, number<alignment>{}, number<1>{}),
                number<alignment>{},
                number<1>{});

            return transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_merge_transform(make_tuple(NumLdsNumA, kKPerBlock)),
                           make_merge_transform(make_tuple(kMPerBlock / alignment, alignment))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumLdsNumA>{},
                           number<kMPerBlock>{},
                           number<kKPerBlock / alignment>{},
                           number<alignment>{}),
                make_tuple(
                    number<kMKPerBlock>{}, number<kKPerBlock>{}, number<alignment>{}, number<1>{}),
                number<alignment>{},
                number<1>{});

            return transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_merge_transform(make_tuple(NumLdsNumA, kMPerBlock)),
                           make_merge_transform(make_tuple(kKPerBlock / alignment, alignment))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock  = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock  = Problem::BlockGemmShape::kK;
        constexpr index_t kNKPerBlock = kNPerBlock * kKPerBlock;

        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;

        constexpr auto alignment = 8;
        if constexpr(is_b_row_major)
        {
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumLdsNumB>{},
                           number<kKPerBlock>{},
                           number<kNPerBlock / alignment>{},
                           number<alignment>{}),
                make_tuple(
                    number<kNKPerBlock>{}, number<kNPerBlock>{}, number<alignment>{}, number<1>{}),
                number<alignment>{},
                number<1>{});

            return transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_merge_transform(make_tuple(NumLdsNumB, kKPerBlock)),
                           make_merge_transform(make_tuple(kNPerBlock / alignment, alignment))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumLdsNumB>{},
                           number<kNPerBlock>{},
                           number<kKPerBlock / alignment>{},
                           number<alignment>{}),
                make_tuple(
                    number<kNKPerBlock>{}, number<kKPerBlock>{}, number<alignment>{}, number<1>{}),
                number<alignment>{},
                number<1>{});

            return transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_merge_transform(make_tuple(NumLdsNumB, kNPerBlock)),
                           make_merge_transform(make_tuple(kKPerBlock / alignment, alignment))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // in wmma TransposeC don't get from Problem::TransposeC
        // TransposeC means if C is Row Major, need to use B * A, need transpose, otherwise if C is
        // column major, use A * B for gfx13; for better performance
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};

        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        using CLayout = remove_cvref_t<typename Problem::CLayout>;
        constexpr bool TransLdA =
            std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>;
        constexpr bool TransLdB = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>;

        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WmmaTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType,
                                            WmmaTile::at(I0),
                                            WmmaTile::at(I1),
                                            WmmaTile::at(I2),
                                            TransposeC,
                                            false,
                                            TransLdA,
                                            TransLdB>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;

        return BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

using GemmPipelineWmmaDefaultPolicy = GemmPipelineWmmaPolicy<false, false, false, 1>;
// Async Copy should enable transpose load
using GemmPipelineWmmaTrLoadAsyncCopyPolicy = GemmPipelineWmmaPolicy<true, true, false, 2>;
// using GemmPipelineWmmaTrLoadAsyncCopyAsyncStorePolicy =
//     GemmPipelineWmmaPolicy<true, true, true, 2, 2>;

} // namespace ck_tile
