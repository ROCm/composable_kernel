// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_fwd_common.hpp"

//#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_preshuffle_multiple_abd_xdl_cshuffle.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using InDataType       = ck::half_t;
using WeiDataType      = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using OutDataType      = ck::half_t;
using F16 = ck::half_t;
using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance = 
    ck::tensor_operation::device::DeviceGroupedConvFwdPreshuffleMultipleABD_Xdl_CShuffle<
    NDimSpatial,
    InLayout,
    WeiLayout,
    ck::Tuple<>,
    OutLayout,   
    F16,   
    F16,     
    F32,      
    F16,    
    ck::Tuple<>,   
    F16, 
    InElementOp, 
    WeiElementOp, 
    OutElementOp,       
    ConvSpec, 
    GemmSpec,        
    1,   
    256,   
    128,    
    64,    
    32,   
    8,   
    8,   
    32,   
    32,    
    2,    
    1,     
    4,    
    64, 3, 3, true, true,      S<1, 4, 4, 8>,     
    S<4, 64, 1>,     
    S<1, 0, 2>,     
    S<1, 0, 2>,              
    2,              
    8,              
    8,         
    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>;
    
    


#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }
