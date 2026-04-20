// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/scale/scale_selector.hpp"
#include "ck_tile/core/arch/mma/scale/scale_transforms.hpp"
#include "ck_tile/core/config.hpp"

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ck_tile::core::arch::mma {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          std::uint32_t FragM,
          std::uint32_t FragN,
          std::uint32_t FragK,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp_ = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          FragM,
                                          FragN,
                                          FragK,
                                          CompilerTarget,
                                          MmaOpFamily::SCALE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct ScaleMmaPipeline : public MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::NONE), ScaleMmaPipeline<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::NONE), ScaleMmaPipeline<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    using MmaOp = MmaOp_; // Expose the selected MmaOp

    // Expose caller-side vector types
    using AVecType = typename MmaOp::AVecType;
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

    template <typename VecTA,
              typename VecTB,
              typename VecTC,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static void
    execImpl(std::tuple<VecTA, VecTB, VecTC, ScaleADataType, ScaleBDataType>& vecs)
    {
        auto& [a_vec, b_vec, c_vec, scale_A, scale_B] = vecs;
        c_vec = MmaOp::exec(a_vec, b_vec, c_vec, scale_A, scale_B);
    }
};

} // namespace ck_tile::core::arch::mma
