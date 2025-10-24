// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "mma.hpp"

namespace ck::tile::core::arch::mma {

/*! @struct Unsupported
 *  @brief  Meta-tag to indicate unsupported amdgcn_mma instance.
 */
struct Unsupported;

/*! @concept MmaOpI
 *  @brief  Expresses the meta-data interface required for each MmaOp policy.
 *  @tparam MmaOp policy class
 */
template <typename MmaOp>
concept MmaOpI = requires(MmaOp op) {
    // Requires an op context
    typename MmaOp::OpType;

    // Captures types for inputs / outputs to mma function
    typename MmaOp::AVecType;
    typename MmaOp::BVecType;
    typename MmaOp::CVecType;

    // Captures CK-specfic layout properties
    { MmaOp::kAMBlock } -> std::convertible_to<unsigned int>;
    { MmaOp::kBNBlock } -> std::convertible_to<unsigned int>;
    { MmaOp::kAMLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kBNLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kABKLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kABKPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCNLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCM0PerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCM1PerLane } -> std::convertible_to<unsigned int>;

    // Static exec function
    {
        MmaOp::exec(
            typename MmaOp::AVecType{}, typename MmaOp::BVecType{}, typename MmaOp::CVecType{})
    } -> std::convertible_to<typename MmaOp::CVecType>;
};

/*! @struct amdgcn_mma
 *  @brief  This is the default MmaOp policy.
 *          Instances of this class are to be used as MmaOp policies.
 *          Light builtin wrapper for mfma / wmma instructions. This class's job is to
 *          provide a uniform interface to invoke the appropriate instruction
 *          based on the template parameters provided. This interface is to bridge
 *          the gap between the ck_tile API types and the native __builtin types.
 *  @tparam DataTypeTA Datatype of input A
 *  @tparam DataTypeTB Datatype of input B
 *  @tparam ComputeT Datatype of accumulator
 *  @tparam BlockM M-dimension of mma block
 *  @tparam BlockN N-dimension of mma block
 *  @tparam BlockK K-dimension of mma block
 *  @tparam CtrlFlags Control flags for mma operation
 *  @tparam GfxTarget The current gfx family target of interest being compiled
 *  @tparam TargetEnable Enabler for the current target if supported
 *  @tparam Enabler SFINAE enabler
 */
template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeAcc,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename CtrlFlags,
          uint32_t GfxTargetId,
          typename Enabler = void>
struct amdgcn_mma
{
    // The base instance is unsupported because there is no __builtin to wrap.
    using OpType = Unsupported;

    // Interface types for A, B, C vectors types
    using AVecType = ext_vector_t<DataTypeA, BlockM * BlockK / get_warp_size()>;
    using BVecType = ext_vector_t<DataTypeB, BlockN * BlockK / get_warp_size()>;
    using CVecType = ext_vector_t<DataTypeAcc, BlockM * BlockN / get_warp_size()>;

    // Layout constants - default to 0
    static constexpr index_t kAMBlock = 0;
    static constexpr index_t kBNBlock = 0;

    static constexpr index_t kAMLane     = 0;
    static constexpr index_t kBNLane     = 0;
    static constexpr index_t kABKLane    = 0;
    static constexpr index_t kABKPerLane = 0;

    static constexpr index_t kCMLane     = 0;
    static constexpr index_t kCNLane     = 0;
    static constexpr index_t kCM0PerLane = 0;
    static constexpr index_t kCM1PerLane = 0;

    // This is a default pass-through implementation that doesn't do anything practical.
    CK_TILE_DEVICE static CVecType const&
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        return regsC; // No-op, just return C
    }
};

} // namespace ck::tile::core::arch::mma
