// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5r1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v6.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v7.hpp"

namespace ck {

enum struct BlockGemmPipelineVersion
{
    v1,
    v2,
    v3,
    v4,
    v5,
    v5r1,
    v6,
    v7,
};

template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
constexpr auto BlockGemmPipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_v1<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
    {
        return BlockwiseGemmXdlops_pipeline_v2<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        return BlockwiseGemmXdlops_pipeline_v3<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
    {
        return BlockwiseGemmXdlops_pipeline_v4<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v5)
    {
        return BlockwiseGemmXdlops_pipeline_v5<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v5r1)
    {
        return BlockwiseGemmXdlops_pipeline_v5r1<BlkGemmPipeSche,
                                                 BlockSize,
                                                 ADataType,
                                                 BDataType,
                                                 ComputeDataType,
                                                 AccDataType,
                                                 ATileDesc,
                                                 BTileDesc,
                                                 AMmaTileDesc,
                                                 BMmaTileDesc,
                                                 ABlockTransferSrcScalarPerVector,
                                                 BBlockTransferSrcScalarPerVector,
                                                 MPerBlock,
                                                 NPerBlock,
                                                 KPerBlock,
                                                 MPerXDL,
                                                 NPerXDL,
                                                 MRepeat,
                                                 NRepeat,
                                                 KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v6)
    {
        return BlockwiseGemmXdlops_pipeline_v6<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v7)
    {
        return BlockwiseGemmXdlops_pipeline_v7<BlkGemmPipeSche,
                                               BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack>{};
    }
    else
    {
        std::cerr << "BlockGemmPipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
