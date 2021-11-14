#ifndef DEVICE_CONV_HPP
#define DEVICE_CONV_HPP

#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceConvFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

struct DeviceConvBwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_in,
                        const void* p_wei,
                        const void* p_out,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

struct DeviceConvWrw : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        void* p_wei,
                        const void* p_out,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

using DeviceConvFwdPtr = std::unique_ptr<DeviceConvFwd>;
using DeviceConvBwdPtr = std::unique_ptr<DeviceConvBwd>;
using DeviceConvWrwPtr = std::unique_ptr<DeviceConvWrw>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
