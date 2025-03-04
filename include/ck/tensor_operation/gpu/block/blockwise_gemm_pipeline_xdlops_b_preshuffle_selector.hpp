// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp"
namespace ck {

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
constexpr auto BlockGemmBPreshufflePipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_bpreshuffle_v1<BlkGemmPipeSche,
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
        return BlockwiseGemmXdlops_pipeline_bpreshuffle_v2<BlkGemmPipeSche,
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
        static_assert(MRepeat >= 4, "MRepeat should at least be 4 in BlockGemmPipelineVersion::v3");
        return BlockwiseGemmXdlops_pipeline_bpreshuffle_v3<BlkGemmPipeSche,
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
