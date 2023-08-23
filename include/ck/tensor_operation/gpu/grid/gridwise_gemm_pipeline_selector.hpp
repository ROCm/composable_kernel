// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <ostream>
#include <string>

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp"

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
        std::cerr << "GridwiseGemmPipeline configuration is not available" << std::endl;
    }
}

std::string getPipelineVersionString(const PipelineVersion& pv)
{
    switch(pv)
    {
    case PipelineVersion::v1: return "PipelineVersion::v1";
    case PipelineVersion::v2: return "PipelineVersion::v2";
    default: return "Unrecognized pipeline version!";
    }
}

} // namespace ck

std::ostream& operator<<(std::ostream& os, const PipelineVersion pv)
{
    os << getPipelineVersionString(pv);
    return os;
}
