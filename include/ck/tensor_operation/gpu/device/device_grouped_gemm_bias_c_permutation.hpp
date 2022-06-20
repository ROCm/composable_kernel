#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmBiasCPermuteDesc
{
    ck::index_t M_, N_, K_;
    ck::index_t stride_A_, stride_B_;

    // hardcoded for 4D permutation
    ck::index_t M0_, M1_, N0_, N1_;
    ck::index_t stride_D_M0_, stride_D_M1_, stride_D_N0_, stride_D_N1_;
    ck::index_t stride_E_M0_, stride_E_M1_, stride_E_N0_, stride_E_N1_;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmBiasCPermute : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<const void*>& p_d,
                        std::vector<void*>& p_c,
                        std::vector<GemmBiasCPermuteDesc>& gemm_transpose_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmBiasCPermutePtr =
    std::unique_ptr<DeviceGroupedGemmBiasCPermute<AElementwiseOperation,
                                                  BElementwiseOperation,
                                                  CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
