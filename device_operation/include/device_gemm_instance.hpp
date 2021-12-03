#ifndef DEVICE_GEMM_INSTANTCE_HPP
#define DEVICE_GEMM_INSTANTCE_HPP

#include "device_gemm.hpp"
#include "element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void add_device_gemm_instance(
    std::vector<DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                              ck::tensor_operation::element_wise::PassThrough,
                              ck::tensor_operation::element_wise::PassThrough>>&);

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
