#pragma once
namespace ck {
namespace profiler {

using DeviceGemmXdlBaseOpPtr =
    std::unique_ptr<ck::tensor_operation::device::DeviceGemmXdlBaseOperator>;

namespace device_gemm_xdl_instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::RowMajor;
using Col = ck::tensor_layout::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void add_device_gemm_xdl_instance(std::vector<DeviceGemmXdlBaseOpPtr>&);

} // namespace device_gemm_xdl_instance
} // namespace profiler
} // namespace ck
