#ifndef DEVICE_CONV_FWD_BIAS_ACTIVATION_HPP
#define DEVICE_CONV_FWD_BIAS_ACTIVATION_HPP

#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConvFwdBiasActivation : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const void* p_bias,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
using DeviceConvFwdBiasActivationPtr =
    std::unique_ptr<DeviceConvFwdBiasActivation<InElementwiseOperation,
                                                WeiElementwiseOperation,
                                                OutElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
