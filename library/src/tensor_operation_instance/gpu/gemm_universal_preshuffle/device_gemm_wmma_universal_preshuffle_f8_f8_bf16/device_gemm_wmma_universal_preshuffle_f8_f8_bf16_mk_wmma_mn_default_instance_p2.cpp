// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn.hpp"

#include <memory>
#include <vector>

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p2(
    std::vector<std::unique_ptr<DeviceGemmV2BPreshuffle<Row,
                                                        Col,
                                                        Row,
                                                        F8,
                                                        F8,
                                                        BF16,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_instances_p2<GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
