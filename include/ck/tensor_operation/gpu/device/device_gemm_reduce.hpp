#pragma once
#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DPtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
struct DeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        DPtrsGlobal p_dxs,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        DxsInElementwiseOperation dxs_in_element_op,
                        DxsReduceAccElementwiseOperation dxs_out_element_op,
                        ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename DPtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
using DeviceGemmReducePtr = std::unique_ptr<DeviceGemmReduce<DPtrsGlobal,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             DxsInElementwiseOperation,
                                                             DxsReduceAccElementwiseOperation>>;

template <typename DPtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename C1ElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
struct DeviceGemmBiasAddReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        const void* p_c0,
                        const void* p_c1,
                        DPtrsGlobal p_dxs,
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
                        C1ElementwiseOperation c1_element_op,
                        DxsInElementwiseOperation dxs_in_element_op,
                        DxsReduceAccElementwiseOperation dxs_out_element_op,
                        ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename DPtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename C1ElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
using DeviceGemmBiasAddReducePtr =
    std::unique_ptr<DeviceGemmBiasAddReduce<DPtrsGlobal,
                                            AElementwiseOperation,
                                            BElementwiseOperation,
                                            CElementwiseOperation,
                                            C1ElementwiseOperation,
                                            DxsInElementwiseOperation,
                                            DxsReduceAccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
