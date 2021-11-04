#ifndef DEVICE_GEMM_XDL_INSTANTCE_HPP
#define DEVICE_GEMM_XDL_INSTANTCE_HPP

#include "device_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_xdl_instance {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void add_device_gemm_xdl_instance(std::vector<DeviceGemmPtr>&);

} // namespace device_gemm_xdl_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
