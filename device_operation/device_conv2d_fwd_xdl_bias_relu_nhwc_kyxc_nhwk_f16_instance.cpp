#include <stdlib.h>
#include "config.hpp"
#include "device_conv2d_fwd_xdl_bias_activation_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_bias_activation_instance {

using F16 = ck::half_t;
using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddRelu     = ck::tensor_operation::element_wise::AddRelu;

using device_conv2d_fwd_xdl_bias_relu_nhwc_kyxc_nhwk_f16_instances = std::tuple<
    // clang-format off
        //################################################################################| InData| WeiData| OutData| AccData|           A|           B|           C| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
        //################################################################################|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
        //################################################################################|       |        |        |        |   Operation|   Operation|   Operation|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
        //################################################################################|       |        |        |        |            |            |            |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   256,   256,   128,     4,  8,   32,   32,    4,    2,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   256,   128,   256,     4,  8,   32,   32,    2,    4,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   128,   128,   128,     4,  8,   32,   32,    4,    2,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   256,   128,   128,     4,  8,   32,   32,    2,    2,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   128,   128,    64,     4,  8,   32,   32,    2,    2,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   128,    64,   128,     4,  8,   32,   32,    2,    2,      S<1, 2, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,    64,    64,    64,     4,  8,   32,   32,    2,    2,      S<1, 4, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   256,   128,    64,     4,  8,   32,   32,    2,    1,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 1, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   256,    64,   128,     4,  8,   32,   32,    1,    2,      S<1, 1, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   128,   128,    32,     4,  8,   32,   32,    2,    1,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 1, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,   128,    32,   128,     4,  8,   32,   32,    1,    2,      S<1, 1, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,    64,    64,    32,     4,  8,   32,   32,    2,    1,      S<1, 4, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>,
        DeviceConv2dFwdXdl_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F16,     F16,     F16,     F32, PassThrough, PassThrough,     AddRelu,    64,    32,    64,     4,  8,   32,   32,    1,    2,      S<1, 2, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>
    // clang-format on
    >;

void add_device_conv2d_fwd_bias_relu_xdl_nhwc_kyxc_nhwk_fp16_instances(
    std::vector<DeviceConvFwdBiasActivationPtr<PassThrough, PassThrough, AddRelu>>&
        instance_container)
{
    using Instances = device_conv2d_fwd_xdl_bias_relu_nhwc_kyxc_nhwk_f16_instances;

    const auto instances = Instances{};

    ck::static_for<0, std::tuple_size_v<Instances>, 1>{}([&](auto i) {
        using Instance = remove_cvref_t<decltype(std::get<i>(instances))>;

        auto instance = Instance{};

        instance_container.push_back(std::make_unique<Instance>(instance));
    });
}

} // namespace device_conv2d_fwd_bias_activation_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
