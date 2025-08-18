// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"

namespace ck_tile::core::arch::mfma
{
    // Define mfma-specific traits such as 
    // - is_mfma_op: is an mfma class of operation
    // - is_mfma_op_supported: has a __builtin instruction backend
    // - amdgcn_mfma_traits: per-instruction implementation details
    // These allow us to operate on generic operations from a more generic Mma layer.
    template <typename MfmaOp>
    struct is_mfma_op : std::false_type {};

    template <typename DataTypeA,
            typename DataTypeB,
            typename ComputeT,
            uint32_t BlockM,
            uint32_t BlockN,
            uint32_t BlockK,
            uint32_t Cbsz,
            uint32_t Abid,
            uint32_t Blgp,
            uint32_t GfxTargetId,
            typename Enabler>
    struct is_mfma_op<
        amdgcn_mfma<DataTypeA, DataTypeB, ComputeT, BlockM, BlockN, BlockK, Cbsz, Abid, Blgp, GfxTargetId, Enabler>
    > : std::true_type {};

    template <typename MfmaOp>
    constexpr static bool is_mfma_op_v = is_mfma_op<MfmaOp>::value;

    // Trait to check if a given MfmaOp is supported (i.e., doesn't have the Unsupported tag)
    template <typename MfmaOp, typename = void>
    struct is_mfma_op_supported : std::true_type {};

    template <typename MfmaOp>
    struct is_mfma_op_supported<
        MfmaOp,
        std::void_t<typename MfmaOp::Unsupported>
    > : std::false_type {};

    template <typename MfmaOp>
    static constexpr bool is_mfma_op_supported_v = is_mfma_op_supported<MfmaOp>::value;

    template<typename MfmaOp>
    struct amdgcn_mfma_traits;

    // Traits struct to store all input template parameters of amdgcn_mfma
    template <typename DataTypeTA_,
            typename DataTypeTB_,
            typename ComputeT_,
            uint32_t BlockM_,
            uint32_t BlockN_,
            uint32_t BlockK_,
            uint32_t Cbsz_,
            uint32_t Abid_,
            uint32_t Blgp_,
            uint32_t GfxTargetId_>
    struct amdgcn_mfma_traits<amdgcn_mfma<DataTypeTA_,
                    DataTypeTB_,
                    ComputeT_,
                    BlockM_,
                    BlockN_,
                    BlockK_,
                    Cbsz_,
                    Abid_,
                    Blgp_,
                    GfxTargetId_>>
    {
        // Template parameters
        using DataTypeTA      = DataTypeTA_;
        using DataTypeTB      = DataTypeTB_;
        using ComputeT        = ComputeT_;
        static constexpr uint32_t BlockM      = BlockM_;
        static constexpr uint32_t BlockN      = BlockN_;
        static constexpr uint32_t BlockK      = BlockK_;
        static constexpr uint32_t Cbsz        = Cbsz_; // Mfma specific
        static constexpr uint32_t Abid        = Abid_; // Mfma specific
        static constexpr uint32_t Blgp        = Blgp_; // Mfma specific
        static constexpr uint32_t GfxTargetId = GfxTargetId_;
        
        // Op specific traits
        using MfmaOp = amdgcn_mfma<DataTypeTA_,
                    DataTypeTB_,
                    ComputeT_,
                    BlockM_,
                    BlockN_,
                    BlockK_,
                    Cbsz_,
                    Abid_,
                    Blgp_,
                    GfxTargetId_>;

        // Common Mma traits, will be required in Mma concept layer.
        constexpr static bool IsSupported = is_mfma_op_supported_v<MfmaOp>;

        using AVecType = typename MfmaOp::AVecType;
        using BVecType = typename MfmaOp::BVecType;
        using CVecType = typename MfmaOp::CVecType;

        static constexpr index_t kAMBlock    = MfmaOp::kAMBlock;
        static constexpr index_t kBNBlock    = MfmaOp::kBNBlock;
        static constexpr index_t kAMLane     = MfmaOp::kAMLane;
        static constexpr index_t kBNLane     = MfmaOp::kBNLane;
        static constexpr index_t kABKLane    = MfmaOp::kABKLane;
        static constexpr index_t kABKPerLane = MfmaOp::kABKPerLane;
        static constexpr index_t kCMLane     = MfmaOp::kCMLane;
        static constexpr index_t kCNLane     = MfmaOp::kCNLane;
        static constexpr index_t kCM0PerLane = MfmaOp::kCM0PerLane;
        static constexpr index_t kCM1PerLane = MfmaOp::kCM1PerLane;

        // Sanity check, this should never fail
        static_assert(is_mfma_op_v<MfmaOp>, "MfmaOp must be a valid amdgcn_mfma type");
    };

} // namespace ck_tile::core::arch::mfma
