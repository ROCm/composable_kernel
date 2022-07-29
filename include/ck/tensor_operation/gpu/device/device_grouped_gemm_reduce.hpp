#pragma once
#include <iostream>
#include "device_base.hpp"
#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename DElementwiseOperation,
          typename ReduceAccDataType>
struct GroupedDeviceGemmSoftmax : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<const void*>& p_a,
                                                              std::vector<const void*>& p_b,
                                                              std::vector<void*>& p_d,
                                                              std::vector<GemmDesc> gemm_shapes,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              DElementwiseOperation d_element_op,
                                                              ReduceAccDataType alpha) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename DElementwiseOperation,
          typename ReduceAccDataType>
using GroupedDeviceGemmSoftmaxPtr = std::unique_ptr<GroupedDeviceGemmSoftmax<AElementwiseOperation,
                                                                             BElementwiseOperation,
                                                                             DElementwiseOperation,
                                                                             ReduceAccDataType>>;

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DPtrsGlobal,
          typename DxsInElementwiseOperation,
          typename DxsAccElementwiseOperation>
struct GroupedDeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<void*>& p_c,
                        std::vector<DPtrsGlobal>& p_ds,
                        std::vector<GemmDesc> gemm_shapes,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        DxsInElementwiseOperation dxs_in_element_op,
                        DxsAccElementwiseOperation dxs_out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DPtrsGlobal,
          typename DxsInElementwiseOperation,
          typename DxsAccElementwiseOperation>
using GroupedDeviceGemmReducePtr =
    std::unique_ptr<GroupedDeviceGemmReduce<AElementwiseOperation,
                                            BElementwiseOperation,
                                            CElementwiseOperation,
                                            DPtrsGlobal,
                                            DxsInElementwiseOperation,
                                            DxsAccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
