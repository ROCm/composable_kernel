// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemm : public DeviceGemmV2<ALayout,
                                        BLayout,
                                        CLayout,
                                        ADataType,
                                        BDataType,
                                        CDataType,
                                        AElementwiseOperation,
                                        BElementwiseOperation,
                                        CElementwiseOperation>
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    protected:
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      ck::index_t M,
                                                      ck::index_t N,
                                                      ck::index_t K,
                                                      ck::index_t StrideA,
                                                      ck::index_t StrideB,
                                                      ck::index_t StrideC,
                                                      ck::index_t, /*KSplit*/
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return MakeArgumentPointer(p_a,
                                   p_b,
                                   p_c,
                                   M,
                                   N,
                                   K,
                                   StrideA,
                                   StrideB,
                                   StrideC,
                                   a_element_op,
                                   b_element_op,
                                   c_element_op);
    }

    bool GetPermuteA() override { return false; }
    bool GetPermuteB() override { return false; }
    index_t GetKPerBlock() override { return 0; }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
