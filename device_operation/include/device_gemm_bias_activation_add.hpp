#ifndef DEVICE_GEMM_BIAS_ACTIVATION_ADD_HPP
#define DEVICE_GEMM_BIAS_ACTIVATION_ADD_HPP

#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmBiasActivationAdd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                              const void* p_b,
                                                              void* p_c,
                                                              const void* p_c0,
                                                              const void* p_c1,
                                                              ck::index_t M,
                                                              ck::index_t N,
                                                              ck::index_t K,
                                                              ck::index_t StrideA,
                                                              ck::index_t StrideB,
                                                              ck::index_t StrideC,
                                                              ck::index_t StrideC1,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGemmBiasActivationAddPtr =
    std::unique_ptr<DeviceGemmBiasActivationAdd<AElementwiseOperation,
                                                BElementwiseOperation,
                                                CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
