// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename AsDataType_,
          typename BsDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename ComputeDataType_ = AsDataType_>
struct GemmPipelineProblemBase
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType       = remove_cvref_t<AsDataType_>;
    using BsDataType       = remove_cvref_t<BsDataType_>;
    using CDataType       = remove_cvref_t<CDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using AsLayout = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout = remove_cvref_t<typename Traits::BsLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool TransposeC = Traits::TransposeC;

    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer = Traits::DoubleSmemBuffer;

    static constexpr auto Scheduler         = GemmPipelineScheduler::Default;
    static constexpr index_t VectorLoadSize = Traits::_VectorSize;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_problem", 
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler);
        // clang-format on
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA(number<I> index)
    {
        using AiDataType = remove_cvref_t<std::tuple_element_t<index.value, AsDataType>>;
        using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;

        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<AiDataType>>::PackedSize;
        if constexpr(std::is_same_v<AiLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(AiDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(AiDataType);
        }
        else
        {
            return VectorLoadSize / sizeof(AiDataType);
        }
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB(number<I> index)
    {
        using BiDataType = remove_cvref_t<std::tuple_element_t<index.value, BsDataType>>;
        using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;

        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BiDataType>>::PackedSize;
        if constexpr(std::is_same_v<BiLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(BiDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(BiDataType);
        }
        else
        {
            return PackedSize * VectorLoadSize / sizeof(BiDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t VectorSizeA(number<I> index) {
        using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
        if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA(index);
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA(index);
        }
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t VectorSizeB(number<I> index) {
        using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
        if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB(index);
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB(index);
        }
    };

    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

// Alias for GemmPipelineProblem
template <typename AsDataType_,
          typename BsDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename ComputeDataType_ = AsDataType_>
using GemmPipelineProblem = GemmPipelineProblemBase<AsDataType_,
                                                    BsDataType_,
                                                    CDataType_,
                                                    BlockGemmShape_,
                                                    Traits_,
                                                    ComputeDataType_>;

template <typename AsDataType_,
          typename BsDataType_,
          typename CDataType_,
          typename AElementwise_,
          typename BElementwise_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename ComputeDataType_        = remove_cvref_t<std::tuple_element_t<0, AsDataType_>>>
struct UniversalGemmPipelineProblem
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType       = remove_cvref_t<AsDataType_>;
    using BsDataType       = remove_cvref_t<BsDataType_>;
    using CDataType       = remove_cvref_t<CDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using AElementwise    = remove_cvref_t<AElementwise_>;
    using BElementwise    = remove_cvref_t<BElementwise_>;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using AsLayout = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout = remove_cvref_t<typename Traits::BsLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer = Traits::DoubleSmemBuffer;

    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;

    static constexpr bool TransposeC            = Traits::TransposeC;
    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;
};

} // namespace ck_tile
