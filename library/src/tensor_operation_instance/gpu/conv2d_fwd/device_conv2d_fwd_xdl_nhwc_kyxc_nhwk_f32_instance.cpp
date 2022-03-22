#include <stdlib.h>
#include "config.hpp"
#include "device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "device_operation_instance.hpp"
#include "host_interface.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_instance {

using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0;

// Compilation parameters for in[n, hi, wi, c] * wei[k, y, x, c] = out[n, ho, wo, k]
using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances = std::tuple<
    // clang-format off
        //################################################################| InData| WeiData| OutData| AccData|          In|         Wei|         Out|    ConvForward| NumDim| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise| Specialization|Spatial|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |        |        |        |   Operation|   Operation|   Operation|               |       |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |        |        |        |            |            |            |               |       |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   256,   256,   128,     4,  4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   256,   128,   256,     4,  4,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   128,   128,   128,     4,  4,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   256,   128,   128,     4,  4,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   128,   128,    64,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   128,    64,   128,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,    64,    64,    64,     4,  4,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   256,   128,    64,     4,  4,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   256,    64,   128,     4,  4,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   128,   128,    32,     4,  4,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,   128,    32,   128,     4,  4,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,    64,    64,    32,     4,  4,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwdDefault,      2,    64,    32,    64,     4,  4,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>
    // clang-format on
    >;

using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_p0_f32_instances = std::tuple<
    // clang-format off
        //################################################################| InData| WeiData| OutData| AccData|          In|         Wei|         Out|    ConvForward| NumDim| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise| Specialization|Spatial|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |        |        |        |   Operation|   Operation|   Operation|               |       |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |        |        |        |            |            |            |               |       |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   256,   256,   128,     4,  4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   256,   128,   256,     4,  4,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   128,   128,   128,     4,  4,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   256,   128,   128,     4,  4,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   128,   128,    64,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   128,    64,   128,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,    64,    64,    64,     4,  4,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   256,   128,    64,     4,  4,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   256,    64,   128,     4,  4,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   128,   128,    32,     4,  4,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,   128,    32,   128,     4,  4,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,    64,    64,    32,     4,  4,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,      2,    64,    32,    64,     4,  4,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>
    // clang-format on
    >;

using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_f32_instances = std::tuple<
    // clang-format off
        //################################################################| InData| WeiData| OutData| AccData|          In|         Wei|         Out|    ConvForward| NumDim| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise| Specialization|Spatial|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |        |        |        |   Operation|   Operation|   Operation|               |       |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |        |        |        |            |            |            |               |       |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   256,   256,   128,     4,  4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   256,   128,   256,     4,  4,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   128,   128,   128,     4,  4,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   256,   128,   128,     4,  4,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   128,   128,    64,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   128,    64,   128,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,    64,    64,    64,     4,  4,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   256,   128,    64,     4,  4,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   256,    64,   128,     4,  4,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   128,   128,    32,     4,  4,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,   128,    32,   128,     4,  4,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,    64,    64,    32,     4,  4,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>,
        DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,      2,    64,    32,    64,     4,  4,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,      true,               7,               1>
    // clang-format on
    >;

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances)
{
    add_device_operation_instances(instances, device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances{});
    add_device_operation_instances(instances,
                                   device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_p0_f32_instances{});
    add_device_operation_instances(instances,
                                   device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_f32_instances{});
}

} // namespace device_conv2d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl
{
    std::unique_ptr<DeviceConvFwdPtr_t::BaseArgument> MakeArgumentPointer(void* in_ptr, void* wei_ptr, void* out_ptr, 
				 size_t N, size_t K, size_t C, 
				 std::vector<ck::index_t> input_spatial_lengths,
				 std::vector<ck::index_t> filter_spatial_lengths,
				 std::vector<ck::index_t> output_spatial_lengths,
				 std::vector<ck::index_t> conv_filter_strides,
				 std::vector<ck::index_t> conv_filter_dilations,
				 std::vector<ck::index_t> input_left_pads,
				 std::vector<ck::index_t> input_right_pads)
    {
        return el->MakeArgumentPointer(in_ptr, wei_ptr, out_ptr, N, K, C, input_spatial_lengths, filter_spatial_lengths, output_spatial_lengths, conv_filter_strides,
        conv_filter_dilations, input_left_pads, input_right_pads, PassThrough{}, PassThrough{}, PassThrough{});
    }
    std::unique_ptr<DeviceConvFwdPtr_t::BaseInvoker> MakeInvokerPointer()
    {
        return el->MakeInvokerPointer();
    }

    std::string GetTypeString()
    {
        return el->GetTypeString();
    }
    bool IsSupportedArgument(const DeviceConvFwdPtr_t::BaseArgument* arg)
    {
        return el->IsSupportedArgument(arg);
    }

    ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough> el;
};

DeviceConvFwdPtr_t::DeviceConvFwdPtr_t() : pImpl(nullptr){}
// DeviceConvFwdPtr_t::DeviceConvFwdPtr_t(DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl& impl) : pImpl(std::make_unique<DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl>(impl)) {}
DeviceConvFwdPtr_t::~DeviceConvFwdPtr_t() = default;
DeviceConvFwdPtr_t::DeviceConvFwdPtr_t(DeviceConvFwdPtr_t&&) = default;
DeviceConvFwdPtr_t::DeviceConvFwdPtr_t(DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl& other) : pImpl(std::make_unique<DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl>(std::move(other))){}

std::unique_ptr<DeviceConvFwdPtr_t::BaseArgument> DeviceConvFwdPtr_t::MakeArgumentPointer(void* in_ptr, void* wei_ptr, void* out_ptr, 
				 size_t N, size_t K, size_t C, 
				 std::vector<ck::index_t> input_spatial_lengths,
				 std::vector<ck::index_t> filter_spatial_lengths,
				 std::vector<ck::index_t> output_spatial_lengths,
				 std::vector<ck::index_t> conv_filter_strides,
				 std::vector<ck::index_t> conv_filter_dilations,
				 std::vector<ck::index_t> input_left_pads,
				 std::vector<ck::index_t> input_right_pads)
{
    return   pImpl->MakeArgumentPointer(in_ptr, wei_ptr, out_ptr, N, K, C, input_spatial_lengths, filter_spatial_lengths, output_spatial_lengths, conv_filter_strides,
        conv_filter_dilations, input_left_pads, input_right_pads);
}

std::unique_ptr<DeviceConvFwdPtr_t::BaseInvoker> DeviceConvFwdPtr_t::MakeInvokerPointer()
{
    return pImpl->MakeInvokerPointer();
}

std::string DeviceConvFwdPtr_t::GetTypeString()
{
    return pImpl->GetTypeString();
}
bool DeviceConvFwdPtr_t::IsSupportedArgument(const DeviceConvFwdPtr_t::BaseArgument* arg_ptr)
{
    return pImpl->IsSupportedArgument(arg_ptr);
}

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances_t(std::vector<DeviceConvFwdPtr_t>& instances)
{
    using namespace ck::tensor_operation::device::device_conv2d_fwd_instance;
    std::vector<ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> local_instances;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(local_instances);
    // convert local_instances to instances
    for(auto& kinder: local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp); // Perhaps we can do better
    }
    return;
}
