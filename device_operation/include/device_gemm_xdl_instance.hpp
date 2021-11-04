#ifndef DEVICE_GEMMXDL_INSTANTCE_HPP
#define DEVICE_GEMMXDL_INSTANTCE_HPP

#include "device_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_xdl_instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

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
