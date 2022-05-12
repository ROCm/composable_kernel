#pragma once

#include <memory>
#include <string>

#include "stream_config.hpp"
#include "config.hpp"
#include "device_base.hpp"

struct DeviceConvFwdPtr_t
{
    using BaseArgument = ck::tensor_operation::device::BaseArgument;
    using BaseInvoker  = ck::tensor_operation::device::BaseInvoker;

    struct DeviceConvFwdPtrImpl;
    std::unique_ptr<DeviceConvFwdPtrImpl> pImpl;
    DeviceConvFwdPtr_t();
    ~DeviceConvFwdPtr_t();
    DeviceConvFwdPtr_t(DeviceConvFwdPtr_t&&);
    DeviceConvFwdPtr_t(DeviceConvFwdPtrImpl&);
    DeviceConvFwdPtr_t& operator=(DeviceConvFwdPtr_t&) = delete;
    DeviceConvFwdPtr_t& operator=(const DeviceConvFwdPtr_t&) = delete;
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* in_ptr,
                        void* wei_ptr,
                        void* out_ptr,
                        size_t N,
                        size_t K,
                        size_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads)
        const; // in,wei and out element ops are ignored for now since even if we change them, they
               // cant be linked
    std::unique_ptr<BaseInvoker>
    MakeInvokerPointer() const; // requires including BaseInvoker headers
    std::string GetTypeString();
    bool IsSupportedArgument(const BaseArgument* arg_ptr);
};

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances);
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances);
