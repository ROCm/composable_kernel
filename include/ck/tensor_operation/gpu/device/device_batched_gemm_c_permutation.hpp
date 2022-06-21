#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemmCPermutation : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                              const void* p_b,
                                                              void* p_c,
                                                              GemmTransposeDesc gemm_transpose_desc,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceBatchedGemmCPermutationPtr =
    std::unique_ptr<DeviceBatchedGemmCPermutation<AElementwiseOperation,
                                                  BElementwiseOperation,
                                                  CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
