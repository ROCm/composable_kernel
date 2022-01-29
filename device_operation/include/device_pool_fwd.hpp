#ifndef DEVICE_POOL_FWD_HPP
#define DEVICE_POOL_FWD_HPP

#include <iostream>
#include <array>
#include "device_base.hpp"
#include "reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InElementwiseOperation, typename AccElementwiseOperation>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        ck::index_t N,
                        ck::index_t C,
                        std::array<ck::index_t, 2> input_spatial_lengths,
                        std::array<ck::index_t, 2> window_spatial_lengths,
                        std::array<ck::index_t, 2> output_spatial_lengths,
                        std::array<ck::index_t, 2> window_strides,
                        std::array<ck::index_t, 2> input_left_pads,
                        std::array<ck::index_t, 2> input_right_pads,
                        const InElementwiseOperation inElementwiseOp,
                        const AccElementwiseOperation accElementwiseOp) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation, typename AccElementwiseOperation>
using DevicePoolFwdPtr =
    std::unique_ptr<DevicePoolFwd<InElementwiseOperation, AccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
