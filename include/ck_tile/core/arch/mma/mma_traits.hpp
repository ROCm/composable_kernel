// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "amdgcn_mma.hpp"
namespace ck::tile::core::arch::mma {
/*! @struct is_mma_op_supported
 * @brief Trait to check if MmaOp is supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
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

/*! @struct MmaOpParams
 * @brief Reflects the template parameters of a given MmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <MmaOpI MmaOp>
struct MmaOpParams;

/*! @concept MmaOpParamsI
 *  @brief  Expresses the required members for each MmaOp
 *  @tparam MmaOp backend policy class
 */
template <typename MmaOpParams>
concept MmaOpParamsI = requires(MmaOpParams op) {
    // Capture template parameters
    typename MmaOpParams::DataTypeA;
    typename MmaOpParams::DataTypeB;
    typename MmaOpParams::DataTypeAcc;
    typename MmaOpParams::CtrlFlags;

    { MmaOpParams::BlockM } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockN } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockK } -> std::convertible_to<unsigned int>;
    { MmaOpParams::GfxTargetId } -> std::convertible_to<unsigned int>;

    // Capture a replicator for the MmaOp
    typename MmaOpParams::template MmaOpReplicator<typename MmaOpParams::DataTypeA,
                                                   typename MmaOpParams::DataTypeB,
                                                   typename MmaOpParams::DataTypeAcc,
                                                   MmaOpParams::BlockM,
                                                   MmaOpParams::BlockN,
                                                   MmaOpParams::BlockK,
                                                   typename MmaOpParams::CtrlFlags,
                                                   MmaOpParams::GfxTargetId>;
};

/*! @struct MmaOpParams
 * @brief Reflects the template parameters of a given MmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename DataTypeA_,
          typename DataTypeB_,
          typename DataTypeAcc_,
          uint32_t BlockM_,
          uint32_t BlockN_,
          uint32_t BlockK_,
          typename CtrlFlags_,
          uint32_t GfxTargetId_>
struct MmaOpParams<amdgcn_mma<DataTypeA_,
                              DataTypeB_,
                              DataTypeAcc_,
                              BlockM_,
                              BlockN_,
                              BlockK_,
                              CtrlFlags_,
                              GfxTargetId_>>
{
    // Capture incoming template parameters
    using DataTypeA                       = DataTypeA_;
    using DataTypeB                       = DataTypeB_;
    using DataTypeAcc                     = DataTypeAcc_;
    static constexpr uint32_t BlockM      = BlockM_;
    static constexpr uint32_t BlockN      = BlockN_;
    static constexpr uint32_t BlockK      = BlockK_;
    using CtrlFlags                       = CtrlFlags_;
    static constexpr uint32_t GfxTargetId = GfxTargetId_;

    // Capture a replicator for the MmaOp, such that we can easily instantiate the same MmaOp
    // with different template parameters
    template <typename DataTypeTA,
              typename DataTypeTB,
              typename DataTypeAcc,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename CtrlFlags,
              uint32_t GfxTargetId>
    using MmaOpReplicator = amdgcn_mma<DataTypeTA,
                                       DataTypeTB,
                                       DataTypeAcc,
                                       BlockM,
                                       BlockN,
                                       BlockK,
                                       CtrlFlags,
                                       GfxTargetId>;
};

/*! @struct MmaOpTraits
 * @brief Reflects the template parameters and static members of a given MmaOp.
 * @tparam MmaOp The matrix multiply-accumulate operation
 */
template <MmaOpI MmaOp>
    requires MmaOpParamsI<MmaOpParams<MmaOp>>
struct MmaOpTraits : public MmaOpParams<MmaOp>
{
    // Capture internal MmaOp static members
    using VecTypeA = typename WmmaOp::VecTypeA;
    using VecTypeB = typename WmmaOp::VecTypeB;
    using VecTypeC = typename WmmaOp::VecTypeC;

    static constexpr index_t kAMBlock    = WmmaOp::kAMBlock;
    static constexpr index_t kBNBlock    = WmmaOp::kBNBlock;
    static constexpr index_t kAMLane     = WmmaOp::kAMLane;
    static constexpr index_t kBNLane     = WmmaOp::kBNLane;
    static constexpr index_t kABKLane    = WmmaOp::kABKLane;
    static constexpr index_t kABKPerLane = WmmaOp::kABKPerLane;
    static constexpr index_t kCMLane     = WmmaOp::kCMLane;
    static constexpr index_t kCNLane     = WmmaOp::kCNLane;
    static constexpr index_t kCM0PerLane = WmmaOp::kCM0PerLane;
    static constexpr index_t kCM1PerLane = WmmaOp::kCM1PerLane;

    // Additional traits to identify the type of MmaOp at compile time
    constexpr static bool IsMfma      = is_mma_op_mfma_v<MmaOp>;
    constexpr static bool IsWmma      = is_mma_op_wmma_v<MmaOp>;
    constexpr static bool IsSupported = is_mma_op_supported_v<MmaOp>;
};

} // namespace ck::tile::core::arch::mma
