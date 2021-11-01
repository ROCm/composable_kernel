#pragma once

namespace ck {
namespace profiler {

using DeviceOpCombo = std::tuple<std::unique_ptr<ck::tensor_operation::device::BaseOperator>,
                                 std::unique_ptr<ck::tensor_operation::device::BaseInvoker>,
                                 std::unique_ptr<ck::tensor_operation::device::BaseArgument>>;

namespace device_gemm_xdl_instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::RowMajor;
using Col = ck::tensor_layout::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

} // namespace device_gemm_xdl_instance
} // namespace profiler
} // namespace ck
