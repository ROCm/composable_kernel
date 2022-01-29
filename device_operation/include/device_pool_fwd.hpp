#ifndef DEVICE_POOL_FWD_HPP
#define DEVICE_POOL_FWD_HPP

#include <iostream>
#include "device_base.hpp"
#include "reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename preUnaryOpType, typename posUnaryOpType>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        void* p_out,
                        ck::index_t N,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> window_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        const preUnaryOpType& preUnaryOp,
                        const posUnaryOpType& posUnaryOp) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename preUnaryOpType, typename posUnaryOpType>
using DevicePoolFwdPtr = std::unique_ptr<DevicePoolFwd<preUnaryOpType, posUnaryOpType>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
