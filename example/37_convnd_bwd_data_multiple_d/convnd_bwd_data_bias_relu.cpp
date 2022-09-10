// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_bwd_data_bias_relu_common.hpp"

#include "ck/tensor_operation/gpu/device/device_convnd_bwd_data_bias_relu_nwc_kxc_nwk_xdl.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InDataType       = ck::half_t;
using WeiDataType      = ck::half_t;
using OutDataType      = ck::half_t;
using DDataType        = ck::half_t;
using CShuffleDataType = ck::half_t;
using AccDataType      = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::AddRelu;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNdBwdDataInstances = std::tuple<
    // clang-format off
                                      //############################################|         Num|     InData|     WeiData|     OutData|     AccData|     CShuffleData|     DData|          In|          Wei|          Out|       ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
                                      //############################################|         Dim|       Type|        Type|        Type|        Type|             Type|      Type| Elementwise|  Elementwise|  Elementwise|               Data|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
                                      //############################################|     Spatial|           |            |            |            |                 |          |   Operation|    Operation|    Operation|     Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
                                      //############################################|            |           |            |            |            |                 |          |            |             |             |                   |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwkBiasActivation_Xdl< NDimSpatial, InDataType, WeiDataType, OutDataType, AccDataType, CShuffleDataType, DDataType, InElementOp, WeiElementOp, OutElementOp, ConvBwdDataDefault,   256,   256,   128,     4,  8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              8,      true, 1,           1,               S<1, 1, 32, 1, 1, 8>,              8>,
        ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwkBiasActivation_Xdl< NDimSpatial, InDataType, WeiDataType, OutDataType, AccDataType, CShuffleDataType, DDataType, InElementOp, WeiElementOp, OutElementOp, ConvBwdDataDefault,   256,   128,   256,     4,  8,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              8,      true, 1,           1,               S<1, 1, 32, 1, 1, 8>,              8>,
        ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwkBiasActivation_Xdl< NDimSpatial, InDataType, WeiDataType, OutDataType, AccDataType, CShuffleDataType, DDataType, InElementOp, WeiElementOp, OutElementOp, ConvBwdDataDefault,   256,   128,   128,     4,  8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              2,              8,      true, 1,           1,               S<1, 1, 32, 1, 1, 8>,              8>,
        ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwkBiasActivation_Xdl< NDimSpatial, InDataType, WeiDataType, OutDataType, AccDataType, CShuffleDataType, DDataType, InElementOp, WeiElementOp, OutElementOp, ConvBwdDataDefault,   128,   128,   128,     4,  8,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              8,      true, 1,           1,               S<1, 1, 32, 1, 1, 8>,              8>,
        ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwkBiasActivation_Xdl< NDimSpatial, InDataType, WeiDataType, OutDataType, AccDataType, CShuffleDataType, DDataType, InElementOp, WeiElementOp, OutElementOp, ConvBwdDataDefault,   128,    32,   128,     4,  8,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 32, 1>,     S<2, 0, 1>,     S<0, 2, 1>,             1,              4,              8,      true, 1,           1,               S<1, 1, 32, 1, 1, 8>,              8>
    // clang-format on
    >;

int main(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv_param{
        2, 1, 128, 256, 256, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        do_verification                   = std::stoi(argv[1]);
        init_method                       = std::stoi(argv[2]);
        time_kernel                       = std::stoi(argv[3]);
        const ck::index_t num_dim_spatial = std::stoi(argv[4]);

        conv_param = ck::utils::conv::parse_conv_param(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    // bias: assume contiguous 1d vector
    auto bias_c_desc =
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(conv_param.C_)}));

    if(conv_param.num_dim_spatial_ == 1)
    {
        using InLayout  = ctc::GNWC;
        using WeiLayout = ctc::GKXC;
        using OutLayout = ctc::GNWK;

        auto device_instances = DeviceConvNdBwdDataInstances<1>{};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck::static_for<0, std::tuple_size_v<DeviceConvNdBwdDataInstances<1>>, 1>{}([&](auto i) {
            const auto device_instance = std::get<i>(device_instances);
            using EachInstance         = ck::remove_cvref_t<decltype(device_instance)>;
            run_conv_bwd_data_bias_relu<1,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementOp,
                                        WeiElementOp,
                                        OutElementOp,
                                        EachInstance>(do_verification,
                                                      init_method,
                                                      time_kernel,
                                                      conv_param,
                                                      in_g_n_c_wis_desc,
                                                      wei_g_k_c_xs_desc,
                                                      out_g_n_k_wos_desc,
                                                      bias_c_desc,
                                                      in_element_op,
                                                      wei_element_op,
                                                      out_element_op);
        });
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        using InLayout  = ctc::GNHWC;
        using WeiLayout = ctc::GKYXC;
        using OutLayout = ctc::GNHWK;

        auto device_instances = DeviceConvNdBwdDataInstances<2>{};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck::static_for<0, std::tuple_size_v<DeviceConvNdBwdDataInstances<2>>, 1>{}([&](auto i) {
            const auto device_instance = std::get<i>(device_instances);
            using EachInstance         = ck::remove_cvref_t<decltype(device_instance)>;
            run_conv_bwd_data_bias_relu<2,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementOp,
                                        WeiElementOp,
                                        OutElementOp,
                                        EachInstance>(do_verification,
                                                      init_method,
                                                      time_kernel,
                                                      conv_param,
                                                      in_g_n_c_wis_desc,
                                                      wei_g_k_c_xs_desc,
                                                      out_g_n_k_wos_desc,
                                                      bias_c_desc,
                                                      in_element_op,
                                                      wei_element_op,
                                                      out_element_op);
        });
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        using InLayout  = ctc::GNDHWC;
        using WeiLayout = ctc::GKZYXC;
        using OutLayout = ctc::GNDHWK;

        auto device_instances = DeviceConvNdBwdDataInstances<3>{};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck::static_for<0, std::tuple_size_v<DeviceConvNdBwdDataInstances<3>>, 1>{}([&](auto i) {
            const auto device_instance = std::get<i>(device_instances);
            using EachInstance         = ck::remove_cvref_t<decltype(device_instance)>;
            run_conv_bwd_data_bias_relu<3,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementOp,
                                        WeiElementOp,
                                        OutElementOp,
                                        EachInstance>(do_verification,
                                                      init_method,
                                                      time_kernel,
                                                      conv_param,
                                                      in_g_n_c_wis_desc,
                                                      wei_g_k_c_xs_desc,
                                                      out_g_n_k_wos_desc,
                                                      bias_c_desc,
                                                      in_element_op,
                                                      wei_element_op,
                                                      out_element_op);
        });
    }

    return 0;
}
