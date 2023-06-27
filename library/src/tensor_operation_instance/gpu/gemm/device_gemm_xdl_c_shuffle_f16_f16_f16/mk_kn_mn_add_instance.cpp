// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    for(const auto& appender :
        DeviceGemm_Xdl_CShuffle_Appenders<Row, Row, Row, F16, F16, F16>::Get())
    {
        appender(instances);
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
