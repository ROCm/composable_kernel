// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType       = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using OutDataType      = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec, // ConvForwardSpecialization
        GemmSpec, // GemmSpecialization
        256,
        256,
        256,
        32,
        8,
        8,
        32,
        32,
        4,
        4,
        S<4, 64, 1>,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        0,
        S<4, 64, 1>,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        0,
        1,
        1,
        S<1, 32, 1, 8>,
        8,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1>;

#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }
