#ifndef REFERENCE_CONV_WRW_HPP
#define REFERENCE_CONV_WRW_HPP

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

        const Tensor<InDataType>& in_n_c_hi_wi_;
        Tensor<WeiDataType>& wei_k_c_y_x_;
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
        using Argument = ReferenceConvBwdWeight::Argument;

        float Run(const Argument& arg)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            auto f_kcyx       = [&](auto k, auto c, auto y, auto x) {
                float v_acc = 0;
                for(int n = 0; n < arg.out_n_k_ho_wo_.mDesc.GetLengths()[0]; ++n)
                {
                    for(int ho = 0; ho < arg.out_n_k_ho_wo_.mDesc.GetLengths()[2]; ++ho)
                    {
                        int hi = ho * arg.conv_strides_[I0] + y * arg.conv_dilations_[I0] -
                                 arg.in_left_pads_[I0];
                        for(int wo = 0; wo < arg.out_n_k_ho_wo_.mDesc.GetLengths()[3]; ++wo)
                        {
                            int wi = wo * arg.conv_strides_[I1] + x * arg.conv_dilations_[I1] -
                                     arg.in_left_pads_[I1];
                            if(hi >= 0 && hi < arg.in_n_c_hi_wi_.mDesc.GetLengths()[2] && wi >= 0 &&
                               wi < arg.in_n_c_hi_wi_.mDesc.GetLengths()[3])
                            {
                                float v_out;
                                float v_in;

                                arg.out_element_op_(
                                    v_out,
                                    ck::type_convert<float>(arg.out_n_k_ho_wo_(n, k, ho, wo)));
                                arg.in_element_op_(
                                    v_in, ck::type_convert<float>(arg.in_n_c_hi_wi_(n, c, hi, wi)));

                                v_acc += v_out * v_in;
                            }
                        }
                    }
                }
                float v_wei;

                arg.wei_element_op_(v_wei, v_acc);

                arg.wei_k_c_y_x_(k, c, y, x) = ck::type_convert<OutDataType>(v_wei);
            };

            make_ParallelTensorFunctor(f_kcyx,
                                       arg.wei_k_c_y_x_.mDesc.GetLengths()[0],
                                       arg.wei_k_c_y_x_.mDesc.GetLengths()[1],
                                       arg.wei_k_c_y_x_.mDesc.GetLengths()[2],
                                       arg.wei_k_c_y_x_.mDesc.GetLengths()[3])(
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
#endif
