// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "mfma/mfma_traits.hpp"
#include "scale/scale_traits.hpp"
#include "sparse/sparse_traits.hpp"
#include "wmma/wmma_traits.hpp"

#include <cstdint>
#include <stdio.h>
#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @class is_mma_op_supported
 * @brief Trait to check if MmaOp is supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp, typename = void>
template <typename MmaOp, typename = void>
struct is_mma_op_supported : std::true_type
{
};

/**
 * @struct is_mma_op_supported
 * @brief The MmaOp is unsupported specialization
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp>
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_supported<MmaOp,
                           std::enable_if_t<std::is_same_v<typename MmaOp::OpType, Unsupported>>>
    : std::false_type
{
};

/**
 * @brief Convenience evaluation of is_mma_op_supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp>
template <typename MmaOp>
static constexpr bool is_mma_op_supported_v = is_mma_op_supported<MmaOp>::value;

template <typename T>
struct MmaOpTraits;

/**
 * @struct MmaOpTraits
 * @brief  Gives additional traits and unexposed template parameters of a given MmaOp
 * @tparam ADataType_ Data type of matrix A
 * @tparam BDataType_ Data type of matrix B
 * @tparam CDataType_ Data type of the accumulator
 * @tparam FragM_ Size of the M dimension
 * @tparam FragN_ Size of the N dimension
 * @tparam FragK_ Size of the K dimension
 * @tparam CtrlFlags_ Control flags for the MMA operation
 * @tparam CompilerTarget_ The compiler target
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t FragM_,
          uint32_t FragN_,
          uint32_t FragK_,
          typename CtrlFlags_,
          typename CompilerTarget_,
          MmaOpFamily OpFamily_>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget_>
struct MmaOpTraits<amdgcn_mma<ADataType_,
                              BDataType_,
                              CDataType_,
                              FragM_,
                              FragN_,
                              FragK_,
                              CtrlFlags_,
                              CompilerTarget_,
                              OpFamily_>>
{
    using MmaOp = amdgcn_mma<ADataType_,
                             BDataType_,
                             CDataType_,
                             FragM_,
                             FragN_,
                             FragK_,
                             CtrlFlags_,
                             CompilerTarget_,
                             OpFamily_>;

    // Capture incoming template parameters not already in amdgcn
    using CtrlFlags      = CtrlFlags_;
    using CompilerTarget = CompilerTarget_;
    // TODO c++20static constexpr amdgcn_target_arch_id GfxTargetId = CompilerTarget_;

    // Additional traits to identify the type of MmaOp at compile time
    constexpr static bool IsMfma   = is_mma_op_mfma_v<MmaOp>;
    constexpr static bool IsWmma   = is_mma_op_wmma_v<MmaOp>;
    constexpr static bool IsDense  = OpFamily_ == MmaOpFamily::DENSE;
    constexpr static bool IsSparse = OpFamily_ == MmaOpFamily::SPARSE;
    constexpr static bool IsScale  = OpFamily_ == MmaOpFamily::SCALE;
    constexpr static bool IsSupported =
        is_mma_op_supported_v<MmaOp> && OpFamily_ != MmaOpFamily::UNDEFINED;
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t FragM_,
          uint32_t FragN_,
          uint32_t FragK_,
          typename CtrlFlags_,
          typename CompilerTarget_,
          MmaOpFamily OpFamily_>
CK_TILE_HOST_DEVICE void print(MmaOpTraits<amdgcn_mma<ADataType_,
                                                      BDataType_,
                                                      CDataType_,
                                                      FragM_,
                                                      FragN_,
                                                      FragK_,
                                                      CtrlFlags_,
                                                      CompilerTarget_,
                                                      OpFamily_>> const& traitsObj)
{
    print(amdgcn_mma<ADataType_,
                     BDataType_,
                     CDataType_,
                     FragM_,
                     FragN_,
                     FragK_,
                     CtrlFlags_,
                     CompilerTarget_,
                     OpFamily_>{});
    printf(
        "Additional     IsMfma / IsWmma          : %d / %d\n", traitsObj.IsMfma, traitsObj.IsWmma);
    printf("               IsDense                  : %d\n", traitsObj.IsDense);
    printf("               IsSparse                 : %d\n", traitsObj.IsSparse);
    printf("               IsScale                  : %d\n", traitsObj.IsScale);
    printf("               IsSupported              : %d\n", traitsObj.IsSupported);
}

} // namespace ck_tile::core::arch::mma
