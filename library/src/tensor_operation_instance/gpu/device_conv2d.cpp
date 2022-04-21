#include <stdlib.h>
#include "config.hpp"
#include "device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "device_operation_instance.hpp"
#include "host_interface.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_instance {
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances);

} // namespace device_conv2d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
struct DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl
{
    std::unique_ptr<DeviceConvFwdPtr_t::BaseArgument>
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
                        std::vector<ck::index_t> input_right_pads) const
    {
        return el->MakeArgumentPointer(in_ptr,
                                       wei_ptr,
                                       out_ptr,
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
                                       PassThrough{},
                                       PassThrough{},
                                       PassThrough{});
    }
    std::unique_ptr<DeviceConvFwdPtr_t::BaseInvoker> MakeInvokerPointer() const
    {
        return el->MakeInvokerPointer();
    }

    std::string GetTypeString() { return el->GetTypeString(); }
    bool IsSupportedArgument(const DeviceConvFwdPtr_t::BaseArgument* arg)
    {
        return el->IsSupportedArgument(arg);
    }

    ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough> el;
};

DeviceConvFwdPtr_t::DeviceConvFwdPtr_t() : pImpl(nullptr) {}
DeviceConvFwdPtr_t::~DeviceConvFwdPtr_t()                    = default;
DeviceConvFwdPtr_t::DeviceConvFwdPtr_t(DeviceConvFwdPtr_t&&) = default;
DeviceConvFwdPtr_t::DeviceConvFwdPtr_t(DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl& other)
    : pImpl(std::make_unique<DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl>(std::move(other)))
{
}

std::unique_ptr<DeviceConvFwdPtr_t::BaseArgument>
DeviceConvFwdPtr_t::MakeArgumentPointer(void* in_ptr,
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
                                        std::vector<ck::index_t> input_right_pads) const
{
    return pImpl->MakeArgumentPointer(in_ptr,
                                      wei_ptr,
                                      out_ptr,
                                      N,
                                      K,
                                      C,
                                      input_spatial_lengths,
                                      filter_spatial_lengths,
                                      output_spatial_lengths,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads);
}

std::unique_ptr<DeviceConvFwdPtr_t::BaseInvoker> DeviceConvFwdPtr_t::MakeInvokerPointer() const
{
    return pImpl->MakeInvokerPointer();
}

std::string DeviceConvFwdPtr_t::GetTypeString() { return pImpl->GetTypeString(); }
bool DeviceConvFwdPtr_t::IsSupportedArgument(const DeviceConvFwdPtr_t::BaseArgument* arg_ptr)
{
    return pImpl->IsSupportedArgument(arg_ptr);
}

using namespace ck::tensor_operation::device::device_conv2d_fwd_instance;
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances)
{
    std::vector<
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>
        local_instances;
    add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(local_instances);
    for(auto& kinder : local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp);
    }
    return;
}

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances)
{
    std::vector<
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>
        local_instances;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(local_instances);
    for(auto& kinder : local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp); // Perhaps we can do better
    }
    return;
}

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances)
{
    std::vector<
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>
        local_instances;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(local_instances);
    for(auto& kinder : local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp); // Perhaps we can do better
    }
    return;
}

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances)
{
    std::vector<
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>
        local_instances;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(local_instances);
    for(auto& kinder : local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp); // Perhaps we can do better
    }
    return;
}

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(
    std::vector<DeviceConvFwdPtr_t>& instances)
{
    std::vector<
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>
        local_instances;
    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(local_instances);
    for(auto& kinder : local_instances)
    {
        DeviceConvFwdPtr_t::DeviceConvFwdPtrImpl tmp{std::move(kinder)};
        instances.emplace_back(tmp);
    }
    return;
}
