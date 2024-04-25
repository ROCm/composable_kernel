// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn_multiply_add_fastgelu_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    ck::Tuple<Row, Row>,
                                                    Row,
                                                    BF16,
                                                    I8,
                                                    ck::Tuple<BF16, BF16>,
                                                    BF16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAddFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn_comp_instances<
            ck::Tuple<Row, Row>,
            ck::Tuple<BF16, BF16>,
            MultiplyAddFastGelu,
            GemmMNKPadding>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_multi_d_bf16_i8_bf16_mk_kn_mn_mem_instances<ck::Tuple<Row, Row>,
                                                                              ck::Tuple<BF16, BF16>,
                                                                              MultiplyAddFastGelu,
                                                                              GemmMNKPadding,
                                                                              Intrawave>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
