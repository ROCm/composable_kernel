// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"
#include "ck/tensor_operation/gpu/device/gemm_dl_algorithm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    GemmSpecialization GemmSpec,
    index_t BlockSize,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t K1,
    index_t M1PerThread,
    index_t N1PerThread,
    index_t KPerThread,
    typename M1N1ThreadClusterM1Xs,
    typename M1N1ThreadClusterN1Xs,
    typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
    typename ABlockTransferSrcVectorTensorContiguousDimOrder,
    typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
    typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
    typename BBlockTransferSrcVectorTensorContiguousDimOrder,
    typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
    typename CThreadTransferSrcDstAccessOrder,
    index_t CThreadTransferSrcDstVectorDim,
    index_t CThreadTransferDstScalarPerVector,
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct DeviceGemmDlDpp8 : public DeviceGemmDl<ADataType,
                                              BDataType,
                                              CDataType,
                                              AccDataType,
                                              ALayout,
                                              BLayout,
                                              CLayout,
                                              AElementwiseOperation,
                                              BElementwiseOperation,
                                              CElementwiseOperation,
                                              GemmSpec,
                                              BlockSize,
                                              MPerBlock,
                                              NPerBlock,
                                              K0PerBlock,
                                              K1,
                                              M1PerThread,
                                              N1PerThread,
                                              KPerThread,
                                              M1N1ThreadClusterM1Xs,
                                              M1N1ThreadClusterN1Xs,
                                              ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                              ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              ABlockTransferSrcAccessOrder,
                                              ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                              ABlockTransferSrcVectorTensorContiguousDimOrder,
                                              ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                              BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                              BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                              BBlockTransferThreadClusterArrangeOrder,
                                              BBlockTransferSrcAccessOrder,
                                              BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                              BBlockTransferSrcVectorTensorContiguousDimOrder,
                                              BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                              CThreadTransferSrcDstAccessOrder,
                                              CThreadTransferSrcDstVectorDim,
                                              CThreadTransferDstScalarPerVector,
                                              GemmDlAlgorithm::Dpp8>

{
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGemmDlDpp8"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << M1PerThread << ", "
            << N1PerThread << ", "
            << KPerThread
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
