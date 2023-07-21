// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Instances = OwnerList<InstanceNT>;

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_default_pipeline_v1_instances(
    Instances&);
void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_default_pipeline_v2_instances(
    Instances&);
void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_interwave_pipeline_v1_instances(
    Instances&);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(Instances& instances)
{
    add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_default_pipeline_v1_instances(
        instances);
    add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_default_pipeline_v2_instances(
        instances);
    add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_1_stage_interwave_pipeline_v1_instances(
        instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
