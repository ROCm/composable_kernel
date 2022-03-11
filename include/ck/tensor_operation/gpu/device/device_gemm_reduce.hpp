#pragma once
#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        void* p_d,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGemmReducePtr = std::unique_ptr<
    DeviceGemmReduce<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
