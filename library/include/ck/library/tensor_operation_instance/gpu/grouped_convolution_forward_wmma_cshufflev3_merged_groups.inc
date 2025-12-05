// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv2d_fwd_wmma_cshufflev3_merged_groups_nhwgc_gkyxc_nhwgk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

// TODO: The XDL version of this function is forward declared but never defined! Oversight in XDL
// implementation?

// void
// add_device_grouped_conv2d_fwd_wmma_cshufflev3_merged_groups_ngchw_gkcyx_ngkhw_int8_instances(
//     std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
//                                                                 NGCHW,
//                                                                 GKCYX,
//                                                                 Empty_Tuple,
//                                                                 NGKHW,
//                                                                 int8_t,
//                                                                 int8_t,
//                                                                 Empty_Tuple,
//                                                                 int8_t,
//                                                                 PassThrough,
//                                                                 PassThrough,
//                                                                 PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
