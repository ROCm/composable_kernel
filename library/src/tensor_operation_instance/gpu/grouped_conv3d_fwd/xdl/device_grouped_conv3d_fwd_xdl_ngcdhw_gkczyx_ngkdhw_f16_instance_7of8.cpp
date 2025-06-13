// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instance.inc"

namespace ck::tensor_operation::device::instance {
template void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances_sharded<8, 6>(
    device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances& instances);
} // namespace ck::tensor_operation::device::instance
