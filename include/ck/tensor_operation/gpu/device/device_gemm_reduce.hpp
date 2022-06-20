#pragma once
#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NumDTensor, ck::index_t NumReduce>
struct DeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        void* p_c,
                        std::array<const void*, NumDTensor> p_ds,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, NumDTensor> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_ops,
                        std::array<void*, NumReduce> reduce_out_element_ops,
                        ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <ck::index_t NumDTensor, ck::index_t NumReduce>
using DeviceGemmReducePtr = std::unique_ptr<DeviceGemmReduce<NumDTensor, NumReduce>>;

// template <typename AElementwiseOperation,
//           typename BElementwiseOperation,
//           typename CElementwiseOperation,
//           typename C1ElementwiseOperation,
//           typename ReduceInElementwiseOperations,
//           typename ReduceAccElementwiseOperations>
// struct DeviceGemmBiasAddReduce : public BaseOperator
// {
//     virtual std::unique_ptr<BaseArgument>
//     MakeArgumentPointer(const void* p_a,
//                         const void* p_b,
//                         void* p_c,
//                         const void* p_c0,
//                         const void* p_c1,
//                         void* p_dxs,
//                         ck::index_t M,
//                         ck::index_t N,
//                         ck::index_t K,
//                         ck::index_t StrideA,
//                         ck::index_t StrideB,
//                         ck::index_t StrideC,
//                         ck::index_t StrideC1,
//                         AElementwiseOperation a_element_op,
//                         BElementwiseOperation b_element_op,
//                         CElementwiseOperation c_element_op,
//                         C1ElementwiseOperation c1_element_op,
//                         ReduceInElementwiseOperations reduce_in_element_ops,
//                         ReduceAccElementwiseOperations reduce_out_element_ops,
//                         ck::index_t BatchCount = 1) = 0;

//     virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
// };

// template <typename AElementwiseOperation,
//           typename BElementwiseOperation,
//           typename CElementwiseOperation,
//           typename C1ElementwiseOperation,
//           typename ReduceInElementwiseOperations,
//           typename ReduceAccElementwiseOperations>
// using DeviceGemmBiasAddReducePtr =
//     std::unique_ptr<DeviceGemmBiasAddReduce<AElementwiseOperation,
//                                             BElementwiseOperation,
//                                             CElementwiseOperation,
//                                             C1ElementwiseOperation,
//                                             ReduceInElementwiseOperations,
//                                             ReduceAccElementwiseOperations>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
