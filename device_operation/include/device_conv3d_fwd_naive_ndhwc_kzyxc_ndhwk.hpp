#ifndef DEVICE_CONV3D_FWD_NAIVE_HPP
#define DEVICE_CONV3D_FWD_NAIVE_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include "convolution_utility.hpp"
#include "device.hpp"
#include "device_conv_fwd.hpp"
#include "common_header.hpp"
#include "naive_conv_fwd.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// specialization for #D conv: in[n, di, hi, wi, c] * wei[k, z, y, x, c] = out[n, do, ho, wo, k]
template <typename InDataType,
          typename WeiDataType, // WeiDataType must be the same as InDataType
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>

{
    using DeviceOp = DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;
    // TODO make A/B datatype different
    using ABDataType = InDataType;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in,
                 const WeiDataType* p_wei,
                 OutDataType* p_out,
                 const index_t N,
                 const index_t K,
                 const index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : N_{N},
              K_{K},
              C_{C},
              in_spatial_lengths_{input_spatial_lengths},
              filter_spatial_lengths_{filter_spatial_lengths},
              out_spatial_lengths_{output_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              p_in_{p_in},
              p_wei_{p_wei},
              p_out_{p_out},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        //  private:
        index_t N_;
        index_t K_;
        index_t C_;
        std::vector<index_t> in_spatial_lengths_;
        std::vector<index_t> filter_spatial_lengths_;
        std::vector<index_t> out_spatial_lengths_;
        std::vector<index_t> conv_filter_strides_;
        std::vector<index_t> conv_filter_dilations_;
        std::vector<index_t> in_left_pads_;
        std::vector<index_t> in_right_pads_;

        const InDataType* p_in_;
        const WeiDataType* p_wei_;
        OutDataType* p_out_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto naive_conv3d_fwd =
                ref::naive_conv_fwd_ndhwc_kzyxc_ndhwk<InDataType,
                                                      WeiDataType,
                                                      OutDataType,
                                                      AccDataType,
                                                      InElementwiseOperation,
                                                      WeiElementwiseOperation,
                                                      OutElementwiseOperation>;

            float ave_time = launch_and_time_kernel(naive_conv3d_fwd,
                                                    nrepeat,
                                                    dim3(256),
                                                    dim3(256),
                                                    0,
                                                    arg.p_in_,
                                                    arg.p_wei_,
                                                    arg.p_out_,
                                                    arg.N_,
                                                    arg.K_,
                                                    arg.C_,
                                                    arg.in_spatial_lengths_[0],
                                                    arg.in_spatial_lengths_[1],
                                                    arg.in_spatial_lengths_[2],
                                                    arg.filter_spatial_lengths_[0],
                                                    arg.filter_spatial_lengths_[1],
                                                    arg.filter_spatial_lengths_[2],
                                                    arg.out_spatial_lengths_[0],
                                                    arg.out_spatial_lengths_[1],
                                                    arg.out_spatial_lengths_[2],
                                                    arg.conv_filter_strides_[0],
                                                    arg.conv_filter_strides_[1],
                                                    arg.conv_filter_strides_[2],
                                                    arg.conv_filter_dilations_[0],
                                                    arg.conv_filter_dilations_[1],
                                                    arg.conv_filter_dilations_[2],
                                                    arg.in_left_pads_[0],
                                                    arg.in_left_pads_[1],
                                                    arg.in_left_pads_[2]);

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        std::vector<index_t> out_spatial_lengths =
            ConvolutionUtility::ComputeOutputSpatialLengths(arg.in_spatial_lengths_,
                                                            arg.filter_spatial_lengths_,
                                                            arg.conv_filter_strides_,
                                                            arg.conv_filter_dilations_,
                                                            arg.in_left_pads_,
                                                            arg.in_right_pads_);

        bool out_lengths_are_consistent = out_spatial_lengths[0] == arg.out_spatial_lengths_[0] &&
                                          out_spatial_lengths[1] == arg.out_spatial_lengths_[1] &&
                                          out_spatial_lengths[2] == arg.out_spatial_lengths_[2];
        return out_lengths_are_consistent;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in,
                             const WeiDataType* p_wei,
                             OutDataType* p_out,
                             const index_t N,
                             const index_t K,
                             const index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_out,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const index_t N,
                        const index_t K,
                        const index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override

    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in),
                                          static_cast<const WeiDataType*>(p_wei),
                                          static_cast<OutDataType*>(p_out),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<>";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif

