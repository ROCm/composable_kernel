// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"

using InDataType   = int8_t;
using WeiDataType  = int8_t;
using BiasDataType = int32_t;
using AccDataType  = int32_t;
using OutDataType  = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using ActivationOp = ck::tensor_operation::element_wise::TanH;
using OutElementOp = ck::tensor_operation::element_wise::Add_Mul_Activation_Mul_Clamp<ActivationOp>;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename BiasLayout,
          typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
        NDimSpatial,
        InDataType,
        WeiDataType,
        ck::Tuple<BiasDataType>,
        OutDataType,
        AccDataType,
        InLayout,
        WeiLayout,
        ck::Tuple<BiasLayout>,
        OutLayout,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,            // ConvForwardSpecialization
        GemmSpec,            // GemmSpecialization
        256,                 // BlockSize
        128,                 // MPerBlock
        128,                 // NPerBlock
        16,                  // K0PerBlock
        4,                   // K1
        4,                   // M1PerThread
        4,                   // N1PerThread
        1,                   // KPerThread
        S<8, 2>,             // M1N1ThreadClusterM1Xs
        S<8, 2>,             // M1N1ThreadClusterN1Xs
        S<8, 1, 1, 4>,       // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
        S<2, 1, 128, 1>,     // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
        S<1, 2, 0, 3>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 2, 0, 3>,       // ABlockTransferSrcAccessOrder
        S<4, 1, 1, 4>,       // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
        S<1, 2, 0, 3>,       // ABlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 4>,       // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
        S<8, 1, 1, 4>,       // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
        S<2, 1, 128, 1>,     // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
        S<1, 2, 0, 3>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 2, 0, 3>,       // BBlockTransferSrcAccessOrder
        S<4, 1, 1, 4>,       // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
        S<1, 2, 0, 3>,       // BBlockTransferSrcVectorTensorContiguousDimOrder
        S<1, 1, 1, 4>,       // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
        S<0, 1, 2, 3, 4, 5>, // CThreadTransferSrcDstAccessOrder
        5,                   // CThreadTransferSrcDstVectorDim
        4>;                  // CThreadTransferDstScalarPerVector

#include "run_conv2d_fwd_bias_perlayer_quantization_example.inc"

int main()
{
    float scale_acc           = 0.5f;
    float scale_z_inv         = 0.5f;
    const auto out_element_op = OutElementOp{scale_z_inv, scale_acc, ActivationOp{}};
    run_conv2d_fwd_bias_perlayer_quantization_example(out_element_op);
}
