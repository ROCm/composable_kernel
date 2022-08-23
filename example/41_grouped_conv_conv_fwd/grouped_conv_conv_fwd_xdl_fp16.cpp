// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_conv_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_conv_fwd_xdl_cshuffle.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using In0DataType       = ck::half_t;
using Wei0DataType      = ck::half_t;
using Acc0DataType      = float;
using Wei1DataType      = ck::half_t;
using Acc1DataType      = float;
using C1ShuffleDataType = float;
using Out1DataType      = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using In0ElementOp  = ck::tensor_operation::element_wise::PassThrough;
using Wei0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Wei1ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Out0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using Out1ElementOp = ck::tensor_operation::element_wise::UnaryConvert;

#if 1
template <ck::index_t NDimSpatial,
          typename In0Layout,
          typename Wei0Layout,
          typename Wei1Layout,
          typename Out1Layout>
struct DeviceGroupedConvConvNDFwdInstance
{
};
#endif

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv0_param{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    ck::utils::conv::ConvParam conv1_param{
        2, 1, 128, 64, 256, {1, 1}, {36, 36}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    const auto in0_element_op  = In0ElementOp{};
    const auto wei0_element_op = Wei0ElementOp{};
    const auto wei1_element_op = Wei1ElementOp{};
    const auto out0_element_op = Out0ElementOp{};
    const auto out1_element_op = Out1ElementOp{};

    const auto run = [&](auto ndim_spatial,
                         auto in0_layout,
                         auto wei0_layout,
                         auto wei1_layout,
                         auto out1_layout) {
        constexpr ck::index_t ndim_spatial_value = ndim_spatial.value;

        using In0Layout  = decltype(in0_layout);
        using Wei0Layout = decltype(wei0_layout);
        using Wei1Layout = decltype(wei1_layout);
        using Out1Layout = decltype(out1_layout);

        const auto in0_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<In0Layout>(
                conv0_param);

        const auto wei0_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Wei0Layout>(
                conv0_param);

        // out0 doesn't physical exist, any layout for host verification is OK
        const auto out0_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Out1Layout>(
                conv0_param);

        const auto wei1_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Wei1Layout>(
                conv1_param);

        const auto out1_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Out1Layout>(
                conv1_param);

        return run_grouped_conv_conv_fwd<ndim_spatial_value,
                                         In0DataType,
                                         Wei0DataType,
                                         Acc0DataType,
                                         Wei1DataType,
                                         Out1DataType,
                                         In0ElementOp,
                                         Wei0ElementOp,
                                         Out0ElementOp,
                                         Wei1ElementOp,
                                         Out1ElementOp,
                                         DeviceGroupedConvConvNDFwdInstance<ndim_spatial_value,
                                                                            In0Layout,
                                                                            Wei0Layout,
                                                                            Wei1Layout,
                                                                            Out1Layout>>(
            do_verification,
            init_method,
            time_kernel,
            conv0_param,
            conv1_param,
            in0_g_n_c_wis_desc,
            wei0_g_k_c_xs_desc,
            out0_g_n_k_wos_desc,
            wei1_g_k_c_xs_desc,
            out1_g_n_k_wos_desc,
            in0_element_op,
            wei0_element_op,
            wei1_element_op,
            out0_element_op,
            out1_element_op);
    };

    namespace ctc = ck::tensor_layout::convolution;

    if(conv0_param.num_dim_spatial_ == 1)
    {
        run(ck::Number<1>{}, ctc::GNWC{}, ctc::GKXC{}, ctc::GKXC{}, ctc::GNWK{});
    }
    else if(conv0_param.num_dim_spatial_ == 2)
    {
        run(ck::Number<2>{}, ctc::GNHWC{}, ctc::GKYXC{}, ctc::GKYXC{}, ctc::GNHWK{});
    }
    else if(conv0_param.num_dim_spatial_ == 3)
    {
        run(ck::Number<3>{}, ctc::GNDHWC{}, ctc::GKZYXC{}, ctc::GKZYXC{}, ctc::GNDHWK{});
    }

    return 0;
}
