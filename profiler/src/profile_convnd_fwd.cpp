// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/conv_util.hpp"
#include "ck/library/utility/fill.hpp"

namespace {

enum struct ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

enum struct ConvDataLayout
{
    NCHW, // 0
    NHWC, // 1
};

namespace ctl = ck::tensor_layout::convolution;

template <int NDim, ConvDataLayout DataLayout>
struct ConvolutionLayouts;

template <>
struct ConvolutionLayouts<1, ConvDataLayout::NHWC>
{
    typedef ctl::NWC Input;
    typedef ctl::KXC Weight;
    typedef ctl::NWK Output;
};
template <>
struct ConvolutionLayouts<2, ConvDataLayout::NHWC>
{
    typedef ctl::NHWC Input;
    typedef ctl::KYXC Weight;
    typedef ctl::NHWK Output;
};
template <>
struct ConvolutionLayouts<3, ConvDataLayout::NHWC>
{
    typedef ctl::NDHWC Input;
    typedef ctl::KZYXC Weight;
    typedef ctl::NDHWK Output;
};
template <>
struct ConvolutionLayouts<1, ConvDataLayout::NCHW>
{
    typedef ctl::NCW Input;
    typedef ctl::KCX Weight;
    typedef ctl::NKW Output;
};
template <>
struct ConvolutionLayouts<2, ConvDataLayout::NCHW>
{
    typedef ctl::NCHW Input;
    typedef ctl::KCYX Weight;
    typedef ctl::NKHW Output;
};
template <>
struct ConvolutionLayouts<3, ConvDataLayout::NCHW>
{
    typedef ctl::NCDHW Input;
    typedef ctl::KCZYX Weight;
    typedef ctl::NKDHW Output;
};

void print_use_msg()
{
    std::cout << "arg1: tensor operation (conv_fwd: ForwardConvolution)\n"
              << "arg2: data type (0: fp32; 1: fp16, 2: bf16, 3: int8)\n"
              << "arg3: data layout (0: NCHW; 1: NHWC)\n"
              << "arg4: verification (0=no, 1=yes)\n"
              << "arg5: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: run kernel # of times (>1)\n"
              << "arg8: N spatial dimensions (default 2)\n"
              << "Following arguments (depending on number of spatial dims):\n"
              << " N, K, C, \n"
              << " <filter spatial dimensions>, (ie Y, X for 2D)\n"
              << " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
              << " <strides>, (ie Sy, Sx for 2D)\n"
              << " <dilations>, (ie Dy, Dx for 2D)\n"
              << " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
              << " <right padding>, (ie RightPy, RightPx for 2D)\n"
              << std::endl;
}

ck::utils::conv::ConvParams parse_params(int num_dim_spatial, int argc, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    int conv_args     = 3 + num_dim_spatial * 6;
    int cmdline_nargs = conv_args + 9;
    if(cmdline_nargs != argc)
    {
        print_use_msg();
        exit(1);
    }
    int arg_idx = 9;

    return ck::utils::conv::parse_conv_params(num_dim_spatial, arg_idx, argv);
}

template <int NDim,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ConvLayouts>
void profile_convnd_instances_impl(const ck::utils::conv::ConvParams& params,
                                   bool do_verification,
                                   bool do_log,
                                   bool time_kernel,
                                   int init_method,
                                   ConvLayouts)
{
    using namespace std::placeholders;
    using namespace ck::utils;

    std::unique_ptr<OpInstance<OutDataType, InDataType, WeiDataType>> conv_instance;

    switch(init_method)
    {
    case 0:
        conv_instance =
            std::make_unique<conv::ConvFwdOpInstance<InDataType,
                                                     WeiDataType,
                                                     OutDataType,
                                                     typename ConvLayouts::Input,
                                                     typename ConvLayouts::Weight,
                                                     typename ConvLayouts::Output>>(params, false);
        break;
    case 1:
        conv_instance = std::make_unique<
            conv::ConvFwdOpInstance<InDataType,
                                    WeiDataType,
                                    OutDataType,
                                    typename ConvLayouts::Input,
                                    typename ConvLayouts::Weight,
                                    typename ConvLayouts::Output,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::utils::FillUniformDistributionIntegerValue<int>,
                                    ck::utils::FillUniformDistributionIntegerValue<int>>>(
            params,
            true,
            ck::utils::FillUniformDistributionIntegerValue<int>{},
            ck::utils::FillUniformDistributionIntegerValue<int>{});
        break;
    case 2:
        conv_instance = std::make_unique<
            conv::ConvFwdOpInstance<InDataType,
                                    WeiDataType,
                                    OutDataType,
                                    typename ConvLayouts::Input,
                                    typename ConvLayouts::Weight,
                                    typename ConvLayouts::Output,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::utils::FillUniformDistribution<InDataType>,
                                    ck::utils::FillUniformDistribution<WeiDataType>>>(
            params,
            true,
            ck::utils::FillUniformDistribution<InDataType>{},
            ck::utils::FillUniformDistribution<WeiDataType>{});
        break;
    default: throw std::runtime_error("Unsupported init method!");
    }

    auto reference_conv_fwd_fun = std::bind(
        conv::run_reference_convolution_forward<NDim, InDataType, WeiDataType, OutDataType>,
        params,
        _1,
        _2,
        _3);

    OpInstanceRunEngine<InDataType, WeiDataType, OutDataType> run_engine(
        *conv_instance, reference_conv_fwd_fun, do_verification);

    auto best_conf = run_engine.Profile(
        conv::ConvolutionFwdInstances<InDataType, WeiDataType, OutDataType>::template Get<NDim>(),
        time_kernel,
        do_verification,
        do_log);

    std::cout << "Best configuration parameters:"
              << "\nname: " << best_conf.best_op_name << "\navg_time: " << best_conf.best_avg_time
              << "\ntflops: " << best_conf.best_tflops << "\nGB/s: " << best_conf.best_gb_per_sec
              << std::endl;
}

template <int NDim>
void profile_convnd_instances(ConvDataType data_type,
                              ConvDataLayout data_layout,
                              const ck::utils::conv::ConvParams& params,
                              bool do_verification,
                              bool do_log,
                              bool time_kernel,
                              int init_method)
{
    switch(data_layout)
    {
    case ConvDataLayout::NHWC: {
        switch(data_type)
        {
        case ConvDataType::F32_F32_F32:
            profile_convnd_instances_impl<NDim, float, float, float>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NHWC>{});
            break;
        case ConvDataType::F16_F16_F16:
            profile_convnd_instances_impl<NDim, ck::half_t, ck::half_t, ck::half_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NHWC>{});
            break;
        case ConvDataType::BF16_BF16_BF16:
            profile_convnd_instances_impl<NDim, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NHWC>{});
            break;
        case ConvDataType::INT8_INT8_INT8:
            profile_convnd_instances_impl<NDim, int8_t, int8_t, int8_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NHWC>{});
            break;
        }
        break;
    }
    case ConvDataLayout::NCHW: {
        switch(data_type)
        {
        case ConvDataType::F32_F32_F32:
            profile_convnd_instances_impl<NDim, float, float, float>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NCHW>{});
            break;
        case ConvDataType::F16_F16_F16:
            profile_convnd_instances_impl<NDim, ck::half_t, ck::half_t, ck::half_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NCHW>{});
            break;
        case ConvDataType::BF16_BF16_BF16:
            profile_convnd_instances_impl<NDim, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NCHW>{});
            break;
        case ConvDataType::INT8_INT8_INT8:
            profile_convnd_instances_impl<NDim, int8_t, int8_t, int8_t>(
                params,
                do_verification,
                do_log,
                time_kernel,
                init_method,
                ConvolutionLayouts<NDim, ConvDataLayout::NCHW>{});
            break;
        }
        break;
    }
    }
}

} // namespace

int profile_convnd_fwd(int argc, char* argv[])
{
    using namespace ck::utils::conv;

    ConvDataType data_type{ConvDataType::F32_F32_F32};
    ConvDataLayout data_layout{ConvDataLayout::NHWC};
    bool do_verification{true};
    int init_method{2};
    bool do_log{false};
    bool time_kernel{false};
    int num_dim_spatial{2};
    ConvParams params;

    if(argc >= 4)
    {
        data_type   = static_cast<ConvDataType>(std::stoi(argv[2]));
        data_layout = static_cast<ConvDataLayout>(std::stoi(argv[3]));
    }
    if(argc >= 9)
    {
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);
        num_dim_spatial = std::stoi(argv[8]);
    }
    if(argc >= 10)
    {
        params = parse_params(num_dim_spatial, argc, argv);
    }

    // TODO Print nice message what is being profiled.

    switch(num_dim_spatial)
    {
    case 1:
        profile_convnd_instances<1>(
            data_type, data_layout, params, do_verification, do_log, time_kernel, init_method);
        break;
    case 2:
        profile_convnd_instances<2>(
            data_type, data_layout, params, do_verification, do_log, time_kernel, init_method);
        break;
    case 3:
        profile_convnd_instances<3>(
            data_type, data_layout, params, do_verification, do_log, time_kernel, init_method);
        break;
    default:
        throw std::runtime_error("profile_conv_fwd: unsupported num_dim_spatial value: " +
                                 std::to_string(num_dim_spatial));
    }

    return 0;
}
