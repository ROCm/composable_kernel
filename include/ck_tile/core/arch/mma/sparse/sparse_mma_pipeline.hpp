// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include <cstdint>
#include <type_traits>

namespace ck_tile::core::arch::mma {

namespace sparse::detail {
// TODO: c++20: return MmaPipelineOptionFlags directly
constexpr inline int getPipelineFlags()
{
    return static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A);
}
} // namespace sparse::detail

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          FragM,
                                          FragN,
                                          FragK,
                                          CompilerTarget,
                                          MmaOpFamily::SPARSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct SparseMmaPipeline : public MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    static_assert(!Base::template hasFlag<MmaPipelineOptionFlag::ABSwap>(),
                  "Cannot transpose C in sparse intrinsics.");

    using MmaOp = MmaOp_; // Expose the selected MmaOp

    // Calculate the uncompressed A vector type
    struct ExternalAVecCalculator
    {
        using AVecTraits               = vector_traits<typename MmaOp::AVecType>;
        static constexpr index_t ASize = AVecTraits::vector_size * MmaOp::kCompressionRatio;
        using AVecType                 = ext_vector_t<typename AVecTraits::scalar_type, ASize>;
    };

    // Expose caller-side vector types
    using AVecType = typename ExternalAVecCalculator::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    // Expose internal vector types
    using InternalAVecT = typename MmaOp::AVecType;
    using InternalBVecT = typename MmaOp::BVecType;
    using InternalCVecT = typename MmaOp::CVecType;

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    template <typename ATransformResult, typename BTransformResult, typename CTransformResult>
    CK_TILE_DEVICE static void
    execImpl(std::tuple<ATransformResult, BTransformResult, CTransformResult>& vecs)
    {
        checkATransformResult<ATransformResult>();
        auto& [a_result, b_vec, c_vec] = vecs;
        auto& [a_vec, idx]             = a_result;
        c_vec                          = MmaOp::exec(a_vec, b_vec, c_vec, idx);
    }

    private:
    // Type check helper - not a device function, so std::declval is available
    template <typename ATransformResult>
    static constexpr void checkATransformResult()
    {
        using ExternalAvecRef = std::add_lvalue_reference_t<AVecType>;
        static_assert(std::is_same_v<ATransformResult,
                                     decltype(ATransform::exec(std::declval<ExternalAvecRef>()))>);
    }
};

} // namespace ck_tile::core::arch::mma
