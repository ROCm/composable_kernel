#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmDesc
{
    ck::index_t M, N, K;
    ck::index_t StrideA, StrideB, StrideC;
};

struct GemmTransposeDesc
{
    ck::index_t M, N, K;
    ck::index_t StrideA, StrideB;

    ck::index_t M0, M1, N0, N1;
    ck::index_t StrideM0, StrideM1, StrideN0, StrideN1;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<const void*>& p_a,
                                                              std::vector<const void*>& p_b,
                                                              std::vector<void*>& p_c,
                                                              std::vector<GemmDesc>& gemm_desc,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmPtr = std::unique_ptr<
    DeviceGroupedGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>>;

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmTranspose : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<void*>& p_c,
                        std::vector<GemmTransposeDesc>& gemm_transpose_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmTransposePtr =
    std::unique_ptr<DeviceGroupedGemmTranspose<AElementwiseOperation,
                                               BElementwiseOperation,
                                               CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
