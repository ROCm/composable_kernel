#ifndef DEVICE_CONV_XDL_INSTANTCE_HPP
#define DEVICE_CONV_XDL_INSTANTCE_HPP

#include "device_conv.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv_xdl_instance {

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_xdl_instance(std::vector<DeviceConvPtr>&);

} // namespace device_conv_xdl_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
