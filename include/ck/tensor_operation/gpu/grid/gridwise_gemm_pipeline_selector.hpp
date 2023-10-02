// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"
#ifndef __HIPCC_RTC__
#include <iostream>
#endif

namespace ck {

enum struct PipelineVersion
{
    v1,
    v2,
};

template <PipelineVersion PipelineVer,
          index_t NumPrefetch     = 1,
          LoopScheduler LoopSched = LoopScheduler::Default>
constexpr auto GridwiseGemmPipeline_Selector()
{
    if constexpr(PipelineVer == PipelineVersion::v1)
    {
        if constexpr(LoopSched == LoopScheduler::Default)
        {
            return GridwiseGemmPipeline_v1<NumPrefetch>{};
        }
        else if constexpr(LoopSched == LoopScheduler::Interwave)
        {
            return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
        }
    }
    else if constexpr(PipelineVer == PipelineVersion::v2)
    {
        return GridwiseGemmPipeline_v2{};
    }
    else
    {
#ifndef __HIPCC_RTC__
        std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
#endif
    }
}

} // namespace ck
