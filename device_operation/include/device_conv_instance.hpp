#ifndef DEVICE_CONV_INSTANTCE_HPP
#define DEVICE_CONV_INSTANTCE_HPP

#include "device_conv.hpp"
#include "element_wise_operation.hpp"

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
void add_device_conv_fwd_instance(
    std::vector<DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough>>&);

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_bwd_instance(
    std::vector<DeviceConvBwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough>>&);

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void add_device_conv_wrw_instance(
    std::vector<DeviceConvWrwPtr<ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough>>&);

} // namespace device_conv_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
