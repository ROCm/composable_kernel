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
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t NumDimSpatial                                                    = 2,
          typename ck::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
struct ReferenceConvBwdData : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(Tensor<InDataType>& input,
                 const Tensor<WeiDataType>& weight,
                 const Tensor<OutDataType>& output,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : input_{input},
              weight_{weight},
              output_{output},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        Tensor<InDataType>& input_;
        const Tensor<WeiDataType>& weight_;
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
        using Argument = ReferenceConvBwdData::Argument;

        float Run(const Argument& arg)
        {
            if constexpr(NumDimSpatial == 1)
            {
                auto f_ncw = [&](auto n, auto c, auto wi) {
                    std::size_t K  = arg.weight_.mDesc.GetLengths()[0];
                    std::size_t X  = arg.weight_.mDesc.GetLengths()[2];
                    std::size_t Wo = arg.output_.mDesc.GetLengths()[2];

                    AccDataType v_acc = 0;

                    for(int x = 0; x < X; ++x)
                    {
                        int w_tmp = wi + arg.in_left_pads_[0] - x * arg.conv_dilations_[0];
                        if(w_tmp % arg.conv_strides_[0] == 0)
                        {
                            int wo = w_tmp / arg.conv_strides_[0];
                            if(wo >= 0 && wo < Wo)
                            {
                                for(int k = 0; k < K; ++k)
                                {
                                    AccDataType v_out = 0;
                                    AccDataType v_wei = 0;

                                    arg.out_element_op_(
                                        v_out,
                                        ck::type_convert<AccDataType>(arg.output_(n, k, wo)));
                                    arg.wei_element_op_(
                                        v_wei, ck::type_convert<AccDataType>(arg.weight_(k, c, x)));

                                    v_acc += v_out * v_wei;
                                }
                            }
                        }
                    }

                    float v_in;
                    arg.in_element_op_(v_in, v_acc);
                    arg.input_(n, c, wi) = ck::type_convert<InDataType>(v_in);
                };

                make_ParallelTensorFunctor(f_ncw,
                                           arg.input_.mDesc.GetLengths()[0],
                                           arg.input_.mDesc.GetLengths()[1],
                                           arg.input_.mDesc.GetLengths()[2])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 2)
            {
                auto f_nchw = [&](auto n, auto c, auto hi, auto wi) {
                    std::size_t K = arg.weight_.mDesc.GetLengths()[0];
                    std::size_t Y = arg.weight_.mDesc.GetLengths()[2];
                    std::size_t X = arg.weight_.mDesc.GetLengths()[3];

                    std::size_t Ho = arg.output_.mDesc.GetLengths()[2];
                    std::size_t Wo = arg.output_.mDesc.GetLengths()[3];

                    AccDataType v_acc = 0;

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
                                    int w_tmp =
                                        wi + arg.in_left_pads_[1] - x * arg.conv_dilations_[1];
                                    if(w_tmp % arg.conv_strides_[1] == 0)
                                    {
                                        int wo = w_tmp / arg.conv_strides_[1];
                                        if(wo >= 0 && wo < Wo)
                                        {
                                            for(int k = 0; k < K; ++k)
                                            {
                                                AccDataType v_out = 0;
                                                AccDataType v_wei = 0;

                                                arg.out_element_op_(v_out,
                                                                    ck::type_convert<AccDataType>(
                                                                        arg.output_(n, k, ho, wo)));
                                                arg.wei_element_op_(v_wei,
                                                                    ck::type_convert<AccDataType>(
                                                                        arg.weight_(k, c, y, x)));

                                                v_acc += v_out * v_wei;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    AccDataType v_in;
                    arg.in_element_op_(v_in, v_acc);
                    arg.input_(n, c, hi, wi) = ck::type_convert<InDataType>(v_in);
                };

                make_ParallelTensorFunctor(f_nchw,
                                           arg.input_.mDesc.GetLengths()[0],
                                           arg.input_.mDesc.GetLengths()[1],
                                           arg.input_.mDesc.GetLengths()[2],
                                           arg.input_.mDesc.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NumDimSpatial == 3)
            {
                auto f_ncdhw = [&](auto n, auto c, auto di, auto hi, auto wi) {
                    std::size_t K = arg.weight_.mDesc.GetLengths()[0];
                    std::size_t Z = arg.weight_.mDesc.GetLengths()[2];
                    std::size_t Y = arg.weight_.mDesc.GetLengths()[3];
                    std::size_t X = arg.weight_.mDesc.GetLengths()[4];

                    std::size_t Do = arg.output_.mDesc.GetLengths()[2];
                    std::size_t Ho = arg.output_.mDesc.GetLengths()[3];
                    std::size_t Wo = arg.output_.mDesc.GetLengths()[4];

                    AccDataType v_acc = 0;

                    for(int z = 0; z < Z; ++z)
                    {
                        int d_tmp = di + arg.in_left_pads_[0] - z * arg.conv_dilations_[0];
                        if(d_tmp % arg.conv_strides_[0] == 0)
                        {
                            int do_ = d_tmp / arg.conv_strides_[0];
                            if(do_ >= 0 && do_ < Do)
                            {
                                for(int y = 0; y < Y; ++y)
                                {
                                    int h_tmp =
                                        hi + arg.in_left_pads_[1] - y * arg.conv_dilations_[1];
                                    if(h_tmp % arg.conv_strides_[1] == 0)
                                    {
                                        int ho = h_tmp / arg.conv_strides_[1];
                                        if(ho >= 0 && ho < Ho)
                                        {
                                            for(int x = 0; x < X; ++x)
                                            {
                                                int w_tmp = wi + arg.in_left_pads_[2] -
                                                            x * arg.conv_dilations_[2];
                                                if(w_tmp % arg.conv_strides_[2] == 0)
                                                {
                                                    int wo = w_tmp / arg.conv_strides_[2];
                                                    if(wo >= 0 && wo < Wo)
                                                    {
                                                        for(int k = 0; k < K; ++k)
                                                        {
                                                            AccDataType v_out = 0;
                                                            AccDataType v_wei = 0;

                                                            arg.out_element_op_(
                                                                v_out,
                                                                ck::type_convert<AccDataType>(
                                                                    arg.output_(
                                                                        n, k, do_, ho, wo)));
                                                            arg.wei_element_op_(
                                                                v_wei,
                                                                ck::type_convert<AccDataType>(
                                                                    arg.weight_(k, c, z, y, x)));

                                                            v_acc += v_out * v_wei;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    AccDataType v_in;
                    arg.in_element_op_(v_in, v_acc);
                    arg.input_(n, c, di, hi, wi) = ck::type_convert<InDataType>(v_in);
                };

                make_ParallelTensorFunctor(f_ncdhw,
                                           arg.input_.mDesc.GetLengths()[0],
                                           arg.input_.mDesc.GetLengths()[1],
                                           arg.input_.mDesc.GetLengths()[2],
                                           arg.input_.mDesc.GetLengths()[3],
                                           arg.input_.mDesc.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
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

    static auto MakeArgument(Tensor<InDataType>& input,
                             const Tensor<WeiDataType>& weight,
                             const Tensor<OutDataType>& output,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{input,
                        weight,
                        output,
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
