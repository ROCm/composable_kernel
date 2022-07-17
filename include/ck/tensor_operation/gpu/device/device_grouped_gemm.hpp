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

    std::vector<ck::index_t> stride_Ds_;
};

template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<std::vector<const void*>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmDesc>& gemm_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmPtr = std::unique_ptr<DeviceGroupedGemm<ALayout,
                                                               BLayout,
                                                               DELayout,
                                                               ADataType,
                                                               BDataType,
                                                               DsDataType,
                                                               EDataType,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
