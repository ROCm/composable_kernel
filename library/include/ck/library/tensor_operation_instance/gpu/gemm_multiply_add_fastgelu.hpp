// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn_multiply_add_fastgelu_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    BF16,
                                                    I8,
                                                    BF16_BF16_Tuple,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAddFastGelu>>>&);

// GEMM + Multiply + Add + FastGelu
template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleD<
    ALayout,
    BLayout,
    ck::Tuple<D0Layout, D1Layout>,
    ELayout,
    ADataType,
    BDataType,
    ck::Tuple<D0DataType, D1DataType>,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::MultiplyAddFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<D0Layout, D1Layout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<D0DataType, D1DataType>,
                                         EDataType,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::MultiplyAddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<D0DataType, bhalf_t> && is_same_v<D1DataType, bhalf_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn_multiply_add_fastgelu_mnkpadding_instances(
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
