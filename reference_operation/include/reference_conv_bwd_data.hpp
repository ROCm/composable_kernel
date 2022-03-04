#ifndef REFERENCE_CONV_BWD_DATA_HPP
#define REFERENCE_CONV_BWD_DATA_HPP

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
          typename OutElementwiseOperation>
struct ReferenceConvBwdData : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(Tensor<InDataType>& in_n_c_hi_wi,
                 const Tensor<WeiDataType>& wei_k_c_y_x,
                 const Tensor<OutDataType>& out_n_k_ho_wo,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : in_n_c_hi_wi_{in_n_c_hi_wi},
              wei_k_c_y_x_{wei_k_c_y_x},
              out_n_k_ho_wo_{out_n_k_ho_wo},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        Tensor<InDataType>& in_n_c_hi_wi_;
        const Tensor<WeiDataType>& wei_k_c_y_x_;
        const Tensor<OutDataType>& out_n_k_ho_wo_;

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
        using Argument = ReferenceConvBwdData::Argument;

        float Run(const Argument& arg)
        {
            auto f_nchw = [&](auto n, auto c, auto hi, auto wi) {
                std::size_t K = arg.wei_k_c_y_x_.mDesc.GetLengths()[0];
                std::size_t Y = arg.wei_k_c_y_x_.mDesc.GetLengths()[2];
                std::size_t X = arg.wei_k_c_y_x_.mDesc.GetLengths()[3];

                std::size_t Ho = arg.out_n_k_ho_wo_.mDesc.GetLengths()[2];
                std::size_t Wo = arg.out_n_k_ho_wo_.mDesc.GetLengths()[3];

                float v_acc = 0;

                for(int y = 0; y < Y; ++y)
                {
                    int h_tmp = hi + arg.in_left_pads_[0] - y * arg.conv_dilations_[0];
                    if(h_tmp % arg.conv_strides_[0] == 0)
                    {
                        int ho = h_tmp / arg.conv_strides_[0];
                        if(ho >= 0 && ho < Ho)
                        {
                            for(int x = 0; x < X; ++x)
                            {
                                int w_tmp = wi + arg.in_left_pads_[1] - x * arg.conv_dilations_[1];
                                if(w_tmp % arg.conv_strides_[1] == 0)
                                {
                                    int wo = w_tmp / arg.conv_strides_[1];
                                    if(wo >= 0 && wo < Wo)
                                    {
                                        for(int k = 0; k < K; ++k)
                                        {
                                            float v_out = 0;
                                            float v_wei = 0;

                                            arg.out_element_op_(
                                                v_out,
                                                ck::type_convert<float>(
                                                    arg.out_n_k_ho_wo_(n, k, ho, wo)));
                                            arg.wei_element_op_(v_wei,
                                                                ck::type_convert<float>(
                                                                    arg.wei_k_c_y_x_(k, c, y, x)));

                                            v_acc += v_out * v_wei;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                float v_in;
                arg.in_element_op_(v_in, v_acc);
                arg.in_n_c_hi_wi_(n, c, hi, wi) = ck::type_convert<InDataType>(v_in);
            };

            make_ParallelTensorFunctor(f_nchw,
                                       arg.in_n_c_hi_wi_.mDesc.GetLengths()[0],
                                       arg.in_n_c_hi_wi_.mDesc.GetLengths()[1],
                                       arg.in_n_c_hi_wi_.mDesc.GetLengths()[2],
                                       arg.in_n_c_hi_wi_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const device::BaseArgument* p_arg, int) override
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

    static auto MakeArgument(Tensor<InDataType>& in_n_c_hi_wi,
                             const Tensor<WeiDataType>& wei_k_c_y_x,
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
        str << "ReferenceConvBwdData"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
#endif
