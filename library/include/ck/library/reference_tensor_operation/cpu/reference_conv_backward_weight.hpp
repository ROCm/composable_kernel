// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename std::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
struct ReferenceConvBwdWeight : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& in_n_c_hi_wi,
                 Tensor<WeiDataType>& wei_k_c_y_x,
                 const Tensor<OutDataType>& out_n_k_ho_wo,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : input_{in_n_c_hi_wi},
              weight_{wei_k_c_y_x},
              output_{out_n_k_ho_wo},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        const Tensor<InDataType>& input_;
        Tensor<WeiDataType>& weight_;
        const Tensor<OutDataType>& output_;

        std::vector<index_t> conv_strides_;
        std::vector<index_t> conv_dilations_;
        std::vector<index_t> in_left_pads_;
        std::vector<index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvBwdWeight::Argument;

        // FIXME: properly implement "TensorView" for doing transpose or refer to dimension by name
        float Run(const Argument& arg)
        {
            // tensor descriptor in NCHW/KXYC/NKHW dimensional order
            HostTensorDescriptor in_desc  = arg.input_.mDesc;
            HostTensorDescriptor wei_desc = arg.weight_.mDesc;
            HostTensorDescriptor out_desc = arg.output_.mDesc;

            // input
            if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NWC>)
            {
                in_desc = transpose_host_tensor_descriptor_given_new2old(
                    in_desc, std::vector<std::size_t>{0, 2, 1});
            }
            else if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NHWC>)
            {
                in_desc = transpose_host_tensor_descriptor_given_new2old(
                    in_desc, std::vector<std::size_t>{0, 3, 1, 2});
            }
            else if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
            {
                in_desc = transpose_host_tensor_descriptor_given_new2old(
                    in_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
            }

            // weight
            if constexpr(is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC>)
            {
                wei_desc = transpose_host_tensor_descriptor_given_new2old(
                    wei_desc, std::vector<std::size_t>{0, 2, 1});
            }
            else if constexpr(is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC>)
            {
                wei_desc = transpose_host_tensor_descriptor_given_new2old(
                    wei_desc, std::vector<std::size_t>{0, 3, 1, 2});
            }
            else if constexpr(is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
            {
                wei_desc = transpose_host_tensor_descriptor_given_new2old(
                    wei_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
            }

            // output
            if constexpr(is_same_v<OutLayout, ck::tensor_layout::convolution::NWK>)
            {
                out_desc = transpose_host_tensor_descriptor_given_new2old(
                    out_desc, std::vector<std::size_t>{0, 2, 1});
            }
            else if constexpr(is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK>)
            {
                out_desc = transpose_host_tensor_descriptor_given_new2old(
                    out_desc, std::vector<std::size_t>{0, 3, 1, 2});
            }
            else if constexpr(is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
            {
                out_desc = transpose_host_tensor_descriptor_given_new2old(
                    out_desc, std::vector<std::size_t>{0, 4, 1, 2, 3});
            }

            if constexpr(NumDimSpatial == 1)
            {
                auto f_kcx = [&](auto k, auto c, auto x) {
                    float v_acc = 0;

                    for(std::size_t n = 0; n < out_desc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t wo = 0; wo < out_desc.GetLengths()[2]; ++wo)
                        {
                            auto wi =
                                ck::type_convert<ck::long_index_t>(wo * arg.conv_strides_[0]) +
                                ck::type_convert<ck::long_index_t>(x * arg.conv_dilations_[0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[0]);

                            if(wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) < in_desc.GetLengths()[2])
                            {
                                float v_out;
                                float v_in;

                                // FIXME hacky
                                arg.out_element_op_(
                                    v_out,
                                    ck::type_convert<float>(
                                        arg.output_
                                            .mData[out_desc.GetOffsetFromMultiIndex(n, k, wo)]));

                                // FIXME hacky
                                arg.in_element_op_(
                                    v_in,
                                    ck::type_convert<float>(
                                        arg.input_
                                            .mData[in_desc.GetOffsetFromMultiIndex(n, c, wi)]));

                                v_acc += v_out * v_in;
                            }
                        }
                    }

                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    // FIXME hacky
                    arg.weight_.mData[wei_desc.GetOffsetFromMultiIndex(k, c, x)] =
                        ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kcx,
                                           wei_desc.GetLengths()[0],
                                           wei_desc.GetLengths()[1],
                                           wei_desc.GetLengths()[2])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 2)
            {
                auto f_kcyx = [&](auto k, auto c, auto y, auto x) {
                    float v_acc = 0;

                    for(std::size_t n = 0; n < out_desc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t ho = 0; ho < out_desc.GetLengths()[2]; ++ho)
                        {
                            auto hi =
                                ck::type_convert<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                ck::type_convert<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[0]);

                            for(std::size_t wo = 0; wo < out_desc.GetLengths()[3]; ++wo)
                            {
                                auto wi =
                                    ck::type_convert<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                    ck::type_convert<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                    ck::type_convert<ck::long_index_t>(arg.in_left_pads_[1]);

                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) < in_desc.GetLengths()[2] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) < in_desc.GetLengths()[3])
                                {
                                    float v_out;
                                    float v_in;

                                    // FIXME hacky
                                    arg.out_element_op_(
                                        v_out,
                                        ck::type_convert<float>(
                                            arg.output_.mData[out_desc.GetOffsetFromMultiIndex(
                                                n, k, ho, wo)]));

                                    // FIXME hacky
                                    arg.in_element_op_(
                                        v_in,
                                        ck::type_convert<float>(
                                            arg.input_.mData[in_desc.GetOffsetFromMultiIndex(
                                                n, c, hi, wi)]));

                                    v_acc += v_out * v_in;
                                }
                            }
                        }
                    }

                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    // FIXME hacky
                    arg.weight_.mData[wei_desc.GetOffsetFromMultiIndex(k, c, y, x)] =
                        ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kcyx,
                                           wei_desc.GetLengths()[0],
                                           wei_desc.GetLengths()[1],
                                           wei_desc.GetLengths()[2],
                                           wei_desc.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 3)
            {
                auto f_kczyx = [&](auto k, auto c, auto z, auto y, auto x) {
                    float v_acc = 0;
                    for(std::size_t n = 0; n < out_desc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t do_ = 0; do_ < out_desc.GetLengths()[2]; ++do_)
                        {
                            auto di =
                                ck::type_convert<ck::long_index_t>(do_ * arg.conv_strides_[0]) +
                                ck::type_convert<ck::long_index_t>(z * arg.conv_dilations_[0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[0]);
                            for(std::size_t ho = 0; ho < out_desc.GetLengths()[3]; ++ho)
                            {
                                auto hi =
                                    ck::type_convert<ck::long_index_t>(ho * arg.conv_strides_[1]) +
                                    ck::type_convert<ck::long_index_t>(y * arg.conv_dilations_[1]) -
                                    ck::type_convert<ck::long_index_t>(arg.in_left_pads_[1]);
                                for(std::size_t wo = 0; wo < out_desc.GetLengths()[4]; ++wo)
                                {
                                    auto wi =
                                        ck::type_convert<ck::long_index_t>(wo *
                                                                           arg.conv_strides_[2]) +
                                        ck::type_convert<ck::long_index_t>(x *
                                                                           arg.conv_dilations_[2]) -
                                        ck::type_convert<ck::long_index_t>(arg.in_left_pads_[2]);

                                    if(di >= 0 &&
                                       ck::type_convert<std::size_t>(di) <
                                           in_desc.GetLengths()[2] &&
                                       hi >= 0 &&
                                       ck::type_convert<std::size_t>(hi) <
                                           in_desc.GetLengths()[3] &&
                                       wi >= 0 &&
                                       ck::type_convert<std::size_t>(wi) < in_desc.GetLengths()[4])
                                    {
                                        float v_out;
                                        float v_in;

                                        // FIXME hacky
                                        arg.out_element_op_(
                                            v_out,
                                            ck::type_convert<float>(
                                                arg.output_.mData[out_desc.GetOffsetFromMultiIndex(
                                                    n, k, do_, ho, wo)]));

                                        // FIXME hacky
                                        arg.in_element_op_(
                                            v_in,
                                            ck::type_convert<float>(
                                                arg.input_.mData[in_desc.GetOffsetFromMultiIndex(
                                                    n, c, di, hi, wi)]));

                                        v_acc += v_out * v_in;
                                    }
                                }
                            }
                        }
                    }

                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    // FIXME hacky
                    arg.weight_.mData[wei_desc.GetOffsetFromMultiIndex(k, c, z, y, x)] =
                        ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kczyx,
                                           wei_desc.GetLengths()[0],
                                           wei_desc.GetLengths()[1],
                                           wei_desc.GetLengths()[2],
                                           wei_desc.GetLengths()[3],
                                           wei_desc.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<InDataType>& in_n_c_hi_wi,
                             Tensor<WeiDataType>& wei_k_c_y_x,
                             const Tensor<OutDataType>& out_n_k_ho_wo,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{in_n_c_hi_wi,
                        wei_k_c_y_x,
                        out_n_k_ho_wo,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceConvBwdWeight"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
