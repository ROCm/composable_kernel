// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_tile_loop.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_multiple_d/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_f16_f16_f16_mk_kn_mn_irregular_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using GemmInstances =
    device_ggemm_md_splitk_xdl_cshuffle_f16_f16_f16_mk_kn_mn_irregular_tile_instances<
        PipelineVersion::v2>;

void add_device_grouped_gemm_multi_d_splitk_cshuffle_f16_f16_f16_mk_kn_mn_irregular_instances_pipeline_v2(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(instances, GemmInstances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
