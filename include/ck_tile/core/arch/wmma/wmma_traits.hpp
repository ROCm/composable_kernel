// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"

namespace ck_tile::core::arch::wmma
{

    // Define wmma-specific traits such as 
    // - is_wmma_op: is an wmma class of operation
    // - is_wmma_op_supported: has a __builtin instruction backend
    // - amdgcn_wmma_traits: per-instruction implementation details
    // These allow us to operate on generic operations from a more generic Mma layer.
    template <typename WmmaOp>
    struct is_wmma_op : std::false_type {};

    template <typename DataTypeA,
            typename DataTypeB,
            typename ComputeT,
            uint32_t BlockM,
            uint32_t BlockN,
            uint32_t BlockK,
            uint32_t GfxTargetId,
            typename Enabler>
    struct is_wmma_op<
        amdgcn_wmma<DataTypeA, DataTypeB, ComputeT, BlockM, BlockN, BlockK, GfxTargetId, Enabler>
    > : std::true_type {};

    template <typename WmmaOp>
    constexpr static bool is_wmma_op_v = is_wmma_op<WmmaOp>::value;

    // Trait to check if a given WmmaOp is supported (i.e., doesn't have the Unsupported tag)
    template <typename WmmaOp, typename = void>
    struct is_wmma_op_supported : std::true_type {};

    template <typename WmmaOp>
    struct is_wmma_op_supported<
        WmmaOp,
        std::void_t<typename WmmaOp::Unsupported>
    > : std::false_type {};

    template <typename WmmaOp>
    static constexpr bool is_wmma_op_supported_v = is_wmma_op_supported<WmmaOp>::value;

    template<typename WmmaOp>
    struct amdgcn_wmma_traits;

    // Traits struct to store all input template parameters of amdgcn_wmma
    template <typename DataTypeTA_,
            typename DataTypeTB_,
            typename ComputeT_,
            uint32_t BlockM_,
            uint32_t BlockN_,
            uint32_t BlockK_,
            uint32_t GfxTargetId_>
    struct amdgcn_wmma_traits<amdgcn_wmma<DataTypeTA_,
                    DataTypeTB_,
                    ComputeT_,
                    BlockM_,
                    BlockN_,
                    BlockK_,
                    GfxTargetId_>>
    {
        // Template parameters
        using DataTypeTA      = DataTypeTA_;
        using DataTypeTB      = DataTypeTB_;
        using ComputeT        = ComputeT_;
        static constexpr uint32_t BlockM      = BlockM_;
        static constexpr uint32_t BlockN      = BlockN_;
        static constexpr uint32_t BlockK      = BlockK_;
        static constexpr uint32_t GfxTargetId = GfxTargetId_;
        
        // Op specific traits
        using WmmaOp = amdgcn_wmma<DataTypeTA_,
                    DataTypeTB_,
                    ComputeT_,
                    BlockM_,
                    BlockN_,
                    BlockK_,
                    GfxTargetId_>;

        // Common Mma traits, will be required in Mma concept layer.
        constexpr static bool IsSupported = is_wmma_op_supported_v<WmmaOp>;

        using AVecType = typename WmmaOp::AVecType;
        using BVecType = typename WmmaOp::BVecType;
        using CVecType = typename WmmaOp::CVecType;

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

        // Sanity check, this should never fail
        static_assert(is_wmma_op_v<WmmaOp>, "WmmaOp must be a valid amdgcn_wmma type");
    };

} // namespace ck_tile::core::arch::wmma
