// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_universal_f16_f16_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_universal_f16_f16_f16_km_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
