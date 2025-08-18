// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

namespace ck::tile::core::arch
{
    namespace detail
    {
        // Forward declaration of the main selector
        template<typename InputTA,
                typename InputTB,
                typename ComputeT,
                uint32_t BlockM,
                uint32_t BlockN,
                uint32_t BlockK,
                uint32_t GfxTargetId = amdgcn_target_arch_id::CURRENT_ARCH_ID,
                typename Enable = void>
        struct MmaSelector;

        // Select based on wmma
        template<typename InputTA,
                typename InputTB,
                typename ComputeT,
                uint32_t BlockM,
                uint32_t BlockN,
                uint32_t BlockK,
                uint32_t GfxTargetId>
        struct MmaSelector<InputTA,
                          InputTB,
                          ComputeT,
                          BlockM,
                          BlockN,
                          BlockK,
                          GfxTargetId,
                          enable_gfx11_gfx12_t<GfxTargetId>> : public WmmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>
        {
        };

        // Select based on mfma
        template<typename InputTA,
                typename InputTB,
                typename ComputeT,
                uint32_t BlockM,
                uint32_t BlockN,
                uint32_t BlockK,
                uint32_t GfxTargetId> // Current max possible K-value for backend instr (most efficient)
        struct MmaSelector<InputTA,
                          InputTB,
                          ComputeT,
                          BlockM,
                          BlockN,
                          BlockK,
                          GfxTargetId,
                          enable_gfx9_t<GfxTargetId>> : public MfmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>
        {
        };

    } // namespace detail

    template<typename InputTA,
            typename InputTB,
            typename ComputeT,
            uint32_t BlockM,
            uint32_t BlockN,
            uint32_t BlockK>
    struct MmaSelector : public detail::MmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>
    {};

} // namespace ck::tile::core::arch
