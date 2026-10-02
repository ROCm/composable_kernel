// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_tdm_kr_ktr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
class BlockFmhaBwdDQDKDVPipelineSelector
{
    static constexpr bool has_dpad1 =
        Problem::Traits::kPadHeadDimQ == 1 || Problem::Traits::kPadHeadDimV == 1;
    static constexpr bool is_decode = Problem::BlockFmhaShape::kMaxSeqLenQ > 0;

    // TDM is a gfx12 instruction, so the decode pipeline only moves onto it for
    // instances the codegen marked. gfx950 also emits decode tiles (two, with
    // max_seq_q 32/16) and they keep the original pipeline.
    static constexpr bool use_tdm_decode = Problem::kUseTdmDecode;

    // The decode pipeline is one type; which of the two staging policies it
    // takes is what separates the TDM form from the plain one.
    using DecodePolicy = std::conditional_t<use_tdm_decode,
                                            BlockFmhaBwdPipelineTrLoadTdmPolicy,
                                            BlockFmhaBwdPipelineTrLoadDefaultPolicy>;

    public:
    template <typename... TS>
    using type_ = std::conditional_t<
        Problem::kUseTrLoad,
        std::conditional_t<is_decode,
                           BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR<TS...>,
                           BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR<TS...>>,
        std::conditional_t<Problem::kUseTdmKRKTR,
                           BlockFmhaBwdDQDKDVPipelineTdmKRKTR<TS...>,
                           std::conditional_t<has_dpad1,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVR<TS...>,
                                              BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<TS...>>>>;
    using type = std::conditional_t<!std::is_same_v<Policy, void>,
                                    type_<Problem, Policy>,
                                    std::conditional_t<Problem::kUseTrLoad && is_decode,
                                                       type_<Problem, DecodePolicy>,
                                                       type_<Problem>>>;
};

template <typename Problem, typename Policy = void>
class BlockFmhaBwdDQDKDVPipeline : public BlockFmhaBwdDQDKDVPipelineSelector<Problem, Policy>::type
{
    public:
    static constexpr const char* name = "auto";
};

} // namespace ck_tile
