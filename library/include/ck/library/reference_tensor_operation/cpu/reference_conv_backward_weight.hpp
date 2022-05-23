#pragma once

#include <iostream>
#include <sstream>
#include "device_base.hpp"
#include "host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// out[N, K, Ho, Wo] = in[N, C, Hi, Wi] * wei[K, C, Y, X]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t NumDimSpatial                                                    = 2,
          typename ck::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
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

        float Run(const Argument& arg)
        {
            if constexpr(NumDimSpatial == 1)
            {
                constexpr auto I0 = Number<0>{};
                auto f_kcx        = [&](auto k, auto c, auto x) {
                    float v_acc = 0;
                    for(std::size_t n = 0; n < arg.output_.mDesc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t wo = 0; wo < arg.output_.mDesc.GetLengths()[2]; ++wo)
                        {
                            auto wi =
                                ck::type_convert<ck::long_index_t>(wo * arg.conv_strides_[I0]) +
                                ck::type_convert<ck::long_index_t>(x * arg.conv_dilations_[I0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I0]);
                            if(wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) < arg.input_.mDesc.GetLengths()[2])
                            {
                                float v_out;
                                float v_in;

                                arg.out_element_op_(v_out,
                                                    ck::type_convert<float>(arg.output_(n, k, wo)));
                                arg.in_element_op_(v_in,
                                                   ck::type_convert<float>(arg.input_(n, c, wi)));

                                v_acc += v_out * v_in;
                            }
                        }
                    }
                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    arg.weight_(k, c, x) = ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kcx,
                                           arg.weight_.mDesc.GetLengths()[0],
                                           arg.weight_.mDesc.GetLengths()[1],
                                           arg.weight_.mDesc.GetLengths()[2])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 2)
            {
                constexpr auto I0 = Number<0>{};
                constexpr auto I1 = Number<1>{};
                auto f_kcyx       = [&](auto k, auto c, auto y, auto x) {
                    float v_acc = 0;
                    for(std::size_t n = 0; n < arg.output_.mDesc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t ho = 0; ho < arg.output_.mDesc.GetLengths()[2]; ++ho)
                        {
                            auto hi =
                                ck::type_convert<ck::long_index_t>(ho * arg.conv_strides_[I0]) +
                                ck::type_convert<ck::long_index_t>(y * arg.conv_dilations_[I0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I0]);
                            for(std::size_t wo = 0; wo < arg.output_.mDesc.GetLengths()[3]; ++wo)
                            {
                                auto wi =
                                    ck::type_convert<ck::long_index_t>(wo * arg.conv_strides_[I1]) +
                                    ck::type_convert<ck::long_index_t>(x *
                                                                       arg.conv_dilations_[I1]) -
                                    ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I1]);
                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) <
                                       arg.input_.mDesc.GetLengths()[2] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) <
                                       arg.input_.mDesc.GetLengths()[3])
                                {
                                    float v_out;
                                    float v_in;

                                    arg.out_element_op_(
                                        v_out, ck::type_convert<float>(arg.output_(n, k, ho, wo)));
                                    arg.in_element_op_(
                                        v_in, ck::type_convert<float>(arg.input_(n, c, hi, wi)));

                                    v_acc += v_out * v_in;
                                }
                            }
                        }
                    }
                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    arg.weight_(k, c, y, x) = ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kcyx,
                                           arg.weight_.mDesc.GetLengths()[0],
                                           arg.weight_.mDesc.GetLengths()[1],
                                           arg.weight_.mDesc.GetLengths()[2],
                                           arg.weight_.mDesc.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 3)
            {
                constexpr auto I0 = Number<0>{};
                constexpr auto I1 = Number<1>{};
                constexpr auto I2 = Number<2>{};
                auto f_kczyx      = [&](auto k, auto c, auto z, auto y, auto x) {
                    float v_acc = 0;
                    for(std::size_t n = 0; n < arg.output_.mDesc.GetLengths()[0]; ++n)
                    {
                        for(std::size_t do_ = 0; do_ < arg.output_.mDesc.GetLengths()[2]; ++do_)
                        {
                            auto di =
                                ck::type_convert<ck::long_index_t>(do_ * arg.conv_strides_[I0]) +
                                ck::type_convert<ck::long_index_t>(z * arg.conv_dilations_[I0]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I0]);
                            for(std::size_t ho = 0; ho < arg.output_.mDesc.GetLengths()[3]; ++ho)
                            {
                                auto hi =
                                    ck::type_convert<ck::long_index_t>(ho * arg.conv_strides_[I1]) +
                                    ck::type_convert<ck::long_index_t>(y *
                                                                       arg.conv_dilations_[I1]) -
                                    ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I1]);
                                for(std::size_t wo = 0; wo < arg.output_.mDesc.GetLengths()[4];
                                    ++wo)
                                {
                                    auto wi =
                                        ck::type_convert<ck::long_index_t>(wo *
                                                                           arg.conv_strides_[I2]) +
                                        ck::type_convert<ck::long_index_t>(
                                            x * arg.conv_dilations_[I2]) -
                                        ck::type_convert<ck::long_index_t>(arg.in_left_pads_[I2]);
                                    if(di >= 0 &&
                                       ck::type_convert<std::size_t>(di) <
                                           arg.input_.mDesc.GetLengths()[2] &&
                                       hi >= 0 &&
                                       ck::type_convert<std::size_t>(hi) <
                                           arg.input_.mDesc.GetLengths()[3] &&
                                       wi >= 0 &&
                                       ck::type_convert<std::size_t>(wi) <
                                           arg.input_.mDesc.GetLengths()[4])
                                    {
                                        float v_out;
                                        float v_in;

                                        arg.out_element_op_(v_out,
                                                            ck::type_convert<float>(
                                                                arg.output_(n, k, do_, ho, wo)));
                                        arg.in_element_op_(
                                            v_in,
                                            ck::type_convert<float>(arg.input_(n, c, di, hi, wi)));

                                        v_acc += v_out * v_in;
                                    }
                                }
                            }
                        }
                    }
                    float v_wei;

                    arg.wei_element_op_(v_wei, v_acc);

                    arg.weight_(k, c, z, y, x) = ck::type_convert<WeiDataType>(v_wei);
                };

                make_ParallelTensorFunctor(f_kczyx,
                                           arg.weight_.mDesc.GetLengths()[0],
                                           arg.weight_.mDesc.GetLengths()[1],
                                           arg.weight_.mDesc.GetLengths()[2],
                                           arg.weight_.mDesc.GetLengths()[3],
                                           arg.weight_.mDesc.GetLengths()[4])(
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
