// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"

namespace ck {

template <index_t PipelineVersion,
          index_t NumPrefetch     = 1,
          LoopScheduler LoopSched = LoopScheduler::Default>
constexpr auto GridwiseGemmPipeline_Selector()
{
    if constexpr(PipelineVersion == 1)
    {
        return GridwiseGemmPipeline_v1_Selector<NumPrefetch, LoopSched>{};
    }
    else if constexpr(PipelineVersion == 2)
    {
        return GridwiseGemmPipeline_v2{};
    }
    else
    {
        // put some error message here
    }
}

} // namespace ck
