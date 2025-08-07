// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"

namespace ck_tile {

template <typename Problem>
class BlockFmhaBwdDQDKDVPipelineSelector
{
    static constexpr bool has_dpad = Problem::Traits::kPadHeadDimQ || Problem::Traits::kPadHeadDimV;

    public:
    using type = std::conditional_t<has_dpad,
                                    BlockFmhaBwdDQDKDVPipelineKRKTRVR<Problem>,
                                    BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<Problem>>;
};

template <typename Problem>
class BlockFmhaBwdDQDKDVPipeline : public BlockFmhaBwdDQDKDVPipelineSelector<Problem>::type
{
    public:
    static constexpr const char* name = "auto";
};

} // namespace ck_tile
