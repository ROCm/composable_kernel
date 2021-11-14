#ifndef DEVICE_CONV_INSTANTCE_HPP
#define DEVICE_CONV_INSTANTCE_HPP

#include "device_conv.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv_instance {

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_fwd_instance(std::vector<DeviceConvFwdPtr>&);

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_bwd_instance(std::vector<DeviceConvBwdPtr>&);

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_wrw_instance(std::vector<DeviceConvWrwPtr>&);

} // namespace device_conv_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
