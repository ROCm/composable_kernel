// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5_default_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV5
{
    static constexpr index_t PrefetchStages  = 3;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;
    static constexpr index_t HotloopUnroll   = 2; // TODO

    CK_TILE_HOST static constexpr bool BlockHasHotLoop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % HotloopUnroll == 1)
        {
            return TailNumber::Odd;
        }
        else
        {
            return TailNumber::Even;
        }
        // TODO
        // if(num_loop % PrefetchStages == 1)
        //{
        //    return TailNumber::Three;
        //}
        // else
        //{
        //    return TailNumber::Two;
        //}
    }
};

/**
 *
 *
 *
 */
template <typename Problem, typename Policy = BaseGemmPipelineAgBgCrCompV5DefaultPolicy>
struct GemmPipelineAgBgCrCompV5 : public BaseGemmPipelineAgBgCrCompV5<Problem>
{
    using Base = BaseGemmPipelineAgBgCrCompV5<Problem>;
}

} // namespace ck_tile
