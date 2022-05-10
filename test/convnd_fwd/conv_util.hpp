#ifndef TEST_CONV_UTIL_HPP
#define TEST_CONV_UTIL_HPP

#include <tuple>

#include "config.hpp"
#include "device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "sequence.hpp"

namespace test {
namespace conv {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<InElementOp, WeiElementOp, OutElementOp>;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t SpatialDims, typename InDataType, typename WeiDataType, typename OutDataType>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         // 
        WeiDataType,        //
        OutDataType,        //
        InDataType,         // 
        InElementOp,        // Input Elementwise Operation
        WeiElementOp,       // Weights Elementwise Operation
        OutElementOp,       // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        SpatialDims,        // SptialDims
        64,                 // BlockSize
        16,                 // MPerBlock
        16,                 // NPerBlock
        4,                  // K0PerBlock
        1,                  // K1                                           
        16,                 // MPerXDL
        16,                 // NPerXDL
        1,                  // MXdlPerWave
        1,                  // NXdlPerWave
        S<1, 16, 1>,        // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,         // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // ABlockTransferSrcAccessOrder
        2,                  // ABlockTransferSrcVectorDim
        1,                  // ABlockTransferSrcScalarPerVector
        1,                  // ABlockTransferDstScalarPerVector_K1
        true,               // ABlockLdsAddExtraM
        S<1, 16, 1>,        // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,         // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // BBlockTransferSrcAccessOrder
        2,                  // BBlockTransferSrcVectorDim
        1,                  // BBlockTransferSrcScalarPerVector
        1,                  // BBlockTransferDstScalarPerVector_K1
        true,               // BBlockTransferAddExtraN
        7,                  // CThreadTransferSrcDstVectorDim
        1>;                 // CThreadTransferDstScalarPerVector
// clang-format on

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
void get_test_convolution_fwd_instance(std::vector<DeviceConvFwdNoOpPtr>& instances)
{
    using ConvInstanceT = DeviceConvNDFwdInstance<NDim, InDataType, WeiDataType, OutDataType>;
    instances.emplace_back(std::make_unique<ConvInstanceT>());
}

} // namespace conv
} // namespace test

#endif
