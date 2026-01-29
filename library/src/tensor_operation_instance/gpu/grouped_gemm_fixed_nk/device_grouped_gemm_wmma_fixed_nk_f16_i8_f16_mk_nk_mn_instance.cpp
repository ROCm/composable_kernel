// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_fixed_nk_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_wmma_fixed_nk_f16_i8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         DsLayout,
                                                         Row,
                                                         F16,
                                                         I8,
                                                         DsDataType,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances)
{
    add_device_grouped_gemm_wmma_fixed_nk_irregular_instances<
        F16,
        I8,
        Row,
        Col,
        device_grouped_gemm_wmma_fixed_nk_mk_nk_mn_irregular_instances>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
