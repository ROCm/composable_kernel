// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename AElementWise_ = ck_tile::element_wise::PassThrough,
          typename BElementWise_ = ck_tile::element_wise::PassThrough,
          typename ComputeDataType_ =
              remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType_>>,
          bool FixedVectorSize_ = false,
          index_t VectorSizeA_  = 1,
          index_t VectorSizeB_  = 1>
struct GemmPipelineProblemBase
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType      = remove_cvref_t<AsDataType_>;
    using BsDataType      = remove_cvref_t<BsDataType_>;
    using EDataType       = remove_cvref_t<EDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    static constexpr bool FixedVectorSize = FixedVectorSize_;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using AsLayout = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout = remove_cvref_t<typename Traits::BsLayout>;
    using ELayout  = remove_cvref_t<typename Traits::ELayout>;

    using AElementWise = remove_cvref_t<AElementWise_>;
    using BElementWise = remove_cvref_t<BElementWise_>;

    using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
    using ALayout   = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayout>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
    using BLayout   = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayout>>;

    static constexpr bool TransposeC = Traits::TransposeC;

    static constexpr index_t NumWaveGroups = Traits::NumWaveGroups;

    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer  = Traits::DoubleSmemBuffer;
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

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(ADataType);
        }
        else
        {
            return VectorLoadSize / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(BDataType);
        }
        else
        {
            return PackedSize * VectorLoadSize / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<ELayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(EDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(EDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeA_;
        }
        else if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();

    static constexpr index_t VectorSizeB = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeB_;
        }
        else if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
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
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename AElementWise_ = ck_tile::element_wise::PassThrough,
          typename BElementWise_ = ck_tile::element_wise::PassThrough,
          typename ComputeDataType_ =
              remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType_>>,
          bool FixedVectorSize_ = false,
          index_t VectorSizeA_  = 1,
          index_t VectorSizeB_  = 1>
using GemmPipelineProblem = GemmPipelineProblemBase<AsDataType_,
                                                    BsDataType_,
                                                    EDataType_,
                                                    BlockGemmShape_,
                                                    Traits_,
                                                    AElementWise_,
                                                    BElementWise_,
                                                    ComputeDataType_,
                                                    FixedVectorSize_,
                                                    VectorSizeA_,
                                                    VectorSizeB_>;

template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename AElementWise_           = ck_tile::element_wise::PassThrough,
          typename BElementWise_           = ck_tile::element_wise::PassThrough,
          typename ComputeDataType_ =
              remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType_>>,
          bool FixedVectorSize_ = false,
          index_t VectorSizeA_  = 1,
          index_t VectorSizeB_  = 1>
struct UniversalGemmPipelineProblem
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType      = remove_cvref_t<AsDataType_>;
    using BsDataType      = remove_cvref_t<BsDataType_>;
    using EDataType       = remove_cvref_t<EDataType_>;
    using AElementWise    = remove_cvref_t<AElementWise_>;
    using BElementWise    = remove_cvref_t<BElementWise_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    static constexpr bool FixedVectorSize = FixedVectorSize_;
    static constexpr index_t VectorSizeA  = VectorSizeA_;
    static constexpr index_t VectorSizeB  = VectorSizeB_;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
    using AsLayout  = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout  = remove_cvref_t<typename Traits::BsLayout>;
    using ELayout   = remove_cvref_t<typename Traits::ELayout>;

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

    static constexpr index_t NumWaveGroups = Traits::NumWaveGroups;
};

} // namespace ck_tile
