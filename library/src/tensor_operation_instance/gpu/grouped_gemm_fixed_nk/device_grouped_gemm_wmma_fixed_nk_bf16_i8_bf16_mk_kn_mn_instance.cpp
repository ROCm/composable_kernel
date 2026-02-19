// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_fixed_nk_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_wmma_fixed_nk_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         DsLayout,
                                                         Row,
                                                         BF16,
                                                         I8,
                                                         DsDataType,
                                                         BF16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances)
{
    add_device_grouped_gemm_wmma_fixed_nk_irregular_instances<
        BF16,
        I8,
        Row,
        Row,
        device_grouped_gemm_wmma_fixed_nk_mk_kn_mn_irregular_instances>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
