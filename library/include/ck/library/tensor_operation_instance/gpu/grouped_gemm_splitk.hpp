// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#ifdef CK_ENABLE_FP16
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmSplitK<Row,
                                                        Col,
                                                        Empty_Tuple,
                                                        Row,
                                                        F16,
                                                        F16,
                                                        Empty_Tuple,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmSplitK<Row,
                                                        Row,
                                                        Empty_Tuple,
                                                        Row,
                                                        F16,
                                                        F16,
                                                        Empty_Tuple,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmSplitK<ALayout,
                                                          BLayout,
                                                          Empty_Tuple,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          Empty_Tuple,
                                                          EDataType,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>
{
    using DeviceOp = DeviceGroupedGemmSplitK<ALayout,
                                             BLayout,
                                             Empty_Tuple,
                                             ELayout,
                                             ADataType,
                                             BDataType,
                                             Empty_Tuple,
                                             EDataType,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
