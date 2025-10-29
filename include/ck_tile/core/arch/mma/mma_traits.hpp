// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "amdgcn_mma.hpp"
#include "mfma/mfma_traits.hpp"
#include "wmma/wmma_traits.hpp"

namespace ck_tile::core::arch::mma {

// /*! @struct is_mma_op_supported
//  * @brief Trait to check if MmaOp is supported
//  * @tparam MmaOp The matrix multiply-accumulate operation type to check
//  */
template <MmaOpI MmaOp, typename = void>
struct is_mma_op_supported : std::true_type
{
};

/*! @struct is_mma_op_supported
 * @brief The MmaOp is unsupported specialization
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <MmaOpI MmaOp>
struct is_mma_op_supported<MmaOp,
                           std::enable_if_t<std::is_same_v<typename MmaOp::OpType, Unsupported>>>
    : std::false_type
{
};

/*! @brief Convenience evaluation of is_mma_op_supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <MmaOpI MmaOp>
static constexpr bool is_mma_op_supported_v = is_mma_op_supported<MmaOp>::value;

// /*! @struct MmaOpParams
//  * @brief Reflects the template parameters of a given MmaOp
//  * @tparam MmaOp The matrix multiply-accumulate operation type to check
//  */
template <MmaOpI MmaOp>
struct MmaOpParams;

// /*! @concept MmaOpParamsI
//  *  @brief  Expresses the required members for each MmaOp
//  *  @tparam MmaOp backend policy class
//  */
template <typename MmaOpParams>
concept MmaOpParamsI = requires(MmaOpParams op) {
    // Capture template parameters
    typename MmaOpParams::ADataType;
    typename MmaOpParams::BDataType;
    typename MmaOpParams::CDataType;
    typename MmaOpParams::CtrlFlags;

    { MmaOpParams::BlockM } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockN } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockK } -> std::convertible_to<unsigned int>;
    { MmaOpParams::GfxTargetId } -> std::convertible_to<unsigned int>;
};

/*! @struct MmaOpParams
 * @brief Reflects the template parameters of a given MmaOp
 * @tparam ADataType_ Data type of matrix A
 * @tparam BDataType_ Data type of matrix B
 * @tparam CDataType_ Data type of the accumulator
 * @tparam BlockM_ Size of the M dimension
 * @tparam BlockN_ Size of the N dimension
 * @tparam BlockK_ Size of the K dimension
 * @tparam CtrlFlags_ Control flags for the MMA operation
 * @tparam GfxTargetId_ Target architecture ID
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t BlockM_,
          uint32_t BlockN_,
          uint32_t BlockK_,
          typename CtrlFlags_,
          uint32_t GfxTargetId_>
struct MmaOpParams<amdgcn_mma<ADataType_,
                              BDataType_,
                              CDataType_,
                              BlockM_,
                              BlockN_,
                              BlockK_,
                              CtrlFlags_,
                              GfxTargetId_>>
{
    // Capture incoming template parameters
    using ADataType                       = ADataType_;
    using BDataType                       = BDataType_;
    using CDataType                       = CDataType_;
    static constexpr uint32_t BlockM      = BlockM_;
    static constexpr uint32_t BlockN      = BlockN_;
    static constexpr uint32_t BlockK      = BlockK_;
    using CtrlFlags                       = CtrlFlags_;
    static constexpr uint32_t GfxTargetId = GfxTargetId_;
};

// /*! @struct MmaOpTraits
//  * @brief Reflects the template parameters and static members of a given MmaOp.
//  * @tparam MmaOp The matrix multiply-accumulate operation
//  */
template <MmaOpI MmaOp>
    requires MmaOpParamsI<MmaOpParams<MmaOp>>
struct MmaOpTraits : public MmaOpParams<MmaOp>
{
    // Capture internal MmaOp static members
    using OpType   = typename MmaOp::OpType;
    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    static constexpr index_t kAMBlock    = MmaOp::kAMBlock;
    static constexpr index_t kBNBlock    = MmaOp::kBNBlock;
    static constexpr index_t kAMLane     = MmaOp::kAMLane;
    static constexpr index_t kBNLane     = MmaOp::kBNLane;
    static constexpr index_t kABKLane    = MmaOp::kABKLane;
    static constexpr index_t kABKPerLane = MmaOp::kABKPerLane;
    static constexpr index_t kCMLane     = MmaOp::kCMLane;
    static constexpr index_t kCNLane     = MmaOp::kCNLane;
    static constexpr index_t kCM0PerLane = MmaOp::kCM0PerLane;
    static constexpr index_t kCM1PerLane = MmaOp::kCM1PerLane;

    // Additional traits to identify the type of MmaOp at compile time
    constexpr static bool IsMfma      = is_mma_op_mfma_v<MmaOp>;
    constexpr static bool IsWmma      = is_mma_op_wmma_v<MmaOp>;
    constexpr static bool IsSupported = is_mma_op_supported_v<MmaOp>;
};

} // namespace ck_tile::core::arch::mma
