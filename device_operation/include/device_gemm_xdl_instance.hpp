#ifndef DEVICE_GEMM_XDL_INSTANCE
#define DEVICE_GEMM_XDL_INSTANCE
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmNoOpPtr = DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough>;

template <>
void add_device_gemm_instance<float,
                              float,
                              float,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<float,
                              float,
                              float,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<float,
                              float,
                              float,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<float,
                              float,
                              float,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<ck::half_t,
                              ck::half_t,
                              ck::half_t,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<ck::half_t,
                              ck::half_t,
                              ck::half_t,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<ck::half_t,
                              ck::half_t,
                              ck::half_t,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

template <>
void add_device_gemm_instance<ck::half_t,
                              ck::half_t,
                              ck::half_t,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::ColumnMajor,
                              ck::tensor_layout::gemm::RowMajor>(std::vector<DeviceGemmNoOpPtr>&);

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif