#pragma once
#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D1ElementwiseOperation>
struct DeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                              const void* p_b,
                                                              void* p_c,
                                                              void* p_d0,
                                                              void* p_d1,
                                                              ck::index_t M,
                                                              ck::index_t N,
                                                              ck::index_t K,
                                                              ck::index_t StrideA,
                                                              ck::index_t StrideB,
                                                              ck::index_t StrideC,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              D1ElementwiseOperation d1_element_op,
                                                              ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D1ElementwiseOperation>
using DeviceGemmReducePtr = std::unique_ptr<DeviceGemmReduce<AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             D1ElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
