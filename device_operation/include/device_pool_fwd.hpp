#ifndef DEVICE_POOL_FWD_HPP
#define DEVICE_POOL_FWD_HPP

#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename OutDataType,
          ReduceTensorOp_t ReduceOp,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const InDataType* p_in,
                        OutDataType* p_out,
                        ck::index_t N,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> window_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        float pad_value,
                        InElementwiseOperation in_element_op,
                        OutElementwiseOperation out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InDataType,
          typename OutDataType,
          ReduceTensorOp_t ReduceOp,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
using DevicePoolFwdPtr = std::unique_ptr<DevicePoolFwd<InDataType,
                                                       OutDataType,
                                                       ReduceOp,
                                                       InElementwiseOperation,
                                                       OutElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
