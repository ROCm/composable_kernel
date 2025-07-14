// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp"

namespace ck {

template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
constexpr auto BlockGemmPipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmWmmaops_pipeline_v1<BlkGemmPipeSche,
                                                BlockSize,
                                                ADataType,
                                                BDataType,
                                                ComputeTypeA,
                                                ComputeTypeB,
                                                AccDataType,
                                                AWmmaTileDesc,
                                                BWmmaTileDesc,
                                                ABlockTransferSrcScalarPerVector,
                                                BBlockTransferSrcScalarPerVector,
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerWmma,
                                                NPerWmma,
                                                MRepeat,
                                                NRepeat,
                                                KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        return BlockwiseGemmWmmaops_pipeline_v3<BlkGemmPipeSche,
                                                BlockSize,
                                                ADataType,
                                                BDataType,
                                                ComputeTypeA,
                                                ComputeTypeB,
                                                AccDataType,
                                                AWmmaTileDesc,
                                                BWmmaTileDesc,
                                                ABlockTransferSrcScalarPerVector,
                                                BBlockTransferSrcScalarPerVector,
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerWmma,
                                                NPerWmma,
                                                MRepeat,
                                                NRepeat,
                                                KPack>{};
    }
    else
    {
        static_assert(false, "BlockGemmPipeline configuration is not available");
    }
}

} // namespace ck
