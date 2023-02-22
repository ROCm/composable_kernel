// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_gnwc_gkxc_gnwk_dl.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightGnwcGkxcGnwk_Dl<
        NDimSpatial,          // NDimSpatial
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        16,                   // K0PerBlock
        2,                    // K1
        4,                    // M1PerThread
        4,                    // N1PerThread
        1,                    // KPerThread
        S<8, 2>,              // M1N1ThreadClusterM1Xs
        S<8, 2>,              // M1N1ThreadClusterN1Xs
        S<1, 8, 1, 1, 2>,     // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
        S<1, 2, 1, 128, 1>,   // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
        S<0, 2, 3, 1, 4>,     // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 3, 1, 4>,     // ABlockTransferSrcAccessOrder
        S<1, 1, 1, 1, 1>,     // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
        S<0, 2, 3, 1, 4>,     // ABlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 1, 1>,     // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
        S<1, 1, 1, 8, 2>,     // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
        S<1, 16, 1, 16, 1>,   // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
        S<0, 1, 4, 2, 3>,     // BBlockTransferThreadClusterArrangeOrder
        S<0, 1, 4, 2, 3>,     // BBlockTransferSrcAccessOrder
        S<1, 1, 1, 8, 1>,     // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
        S<0, 1, 4, 2, 3>,     // BBlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 1, 2>,     // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
        S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
        5,                    // CThreadTransferSrcDstVectorDim
        4>;                   // CThreadTransferDstScalarPerVector

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_bwd_weight_example(argc, argv); }
