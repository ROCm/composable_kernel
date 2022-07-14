#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmDesc
{
    ck::index_t M_, N_, K_;
    ck::index_t stride_A_, stride_B_, stride_C_;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<void*>& p_c,
                        std::vector<GemmDesc>& gemm_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmPtr = std::unique_ptr<DeviceGroupedGemm<ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               ADataType,
                                                               BDataType,
                                                               CDataType,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
