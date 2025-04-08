// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp"

namespace ck {

/**
 * @brief Define matrix data types that have hardware support for MX GEMMs
 */
template <typename T>
static constexpr bool is_scale_mfma_data_type()
{
    return is_same_v<T, f8_ocp_t> || is_same_v<T, bf8_ocp_t> || is_same_v<T, f6_t> ||
           is_same_v<T, bf6_t> || is_same_v<T, f4_t>;
}

/**
 * @brief Define scale data types that have hardware support for MX GEMMs
 */
template <typename T>
static constexpr bool is_scale_mfma_scale_type()
{
    return is_same_v<T, e8m0_bexp_t>;
}

/**
 * @brief Combination of data types that have hardware support for MX GEMMs
 */
template <typename ADataType, typename BDataType, typename AScaleDataType, typename BScaleDataType>
static constexpr bool scale_mfma_hw_support()
{
    return is_scale_mfma_data_type<ADataType>() && is_scale_mfma_data_type<BDataType>() &&
           is_scale_mfma_scale_type<AScaleDataType>() && is_scale_mfma_scale_type<BScaleDataType>();
}

template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ComputeDataType, // TODO: remove this as in this pipeline ADataType and BDataType
                                    // must be used for compute
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
constexpr auto BlockGemmMXPipeline_Selector()
{

    // Hardware MX GEMM pipeline
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_v1_mx<BlkGemmPipeSche,
                                                  ThreadBlockSize,
                                                  ScaleBlockSize,
                                                  ADataType,
                                                  AScaleDataType,
                                                  BDataType,
                                                  BScaleDataType,
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
        std::cerr << "MX GEMM Pipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
