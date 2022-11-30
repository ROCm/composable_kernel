// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"

namespace ck {

enum struct PipelineVersion
{
    v1,
    v1_opt0,
    v2,
};

inline std::ostream& operator<<(std::ostream& stream, PipelineVersion version)
{
    switch(version)
    {
    case PipelineVersion::v1: return stream << "v1";
    case PipelineVersion::v1_opt0: return stream << "v1_opt0";
    case PipelineVersion::v2: return stream << "v2";
    }

    __builtin_unreachable();
}

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
    else if constexpr(PipelineVer == PipelineVersion::v1_opt0)
    {
        return GridwiseGemmPipeline_v1<NumPrefetch, 0>{};
    }
    else if constexpr(PipelineVer == PipelineVersion::v2)
    {
        return GridwiseGemmPipeline_v2{};
    }
    else
    {
        std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
