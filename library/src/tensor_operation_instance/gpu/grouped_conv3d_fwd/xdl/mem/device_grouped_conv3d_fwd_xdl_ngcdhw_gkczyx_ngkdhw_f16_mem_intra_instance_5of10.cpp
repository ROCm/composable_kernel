// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_intra_instance.inc"

namespace ck::tensor_operation::device::instance {
template
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_intra_instances_sharded<10, 4>(
    device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_mem_intra_instances& instances);
} // namespace ck::tensor_operation::device::instance
