# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NHWGC layout instances for grouped_conv2d_fwd_dynamic_op
set(GROUPED_CONV2D_FWD_DYNAMIC_OP_NHWGC
   # wmma
   wmma/device_grouped_conv2d_fwd_wmma_dynamic_op_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_dynamic_op_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   # xdl
   xdl/device_grouped_conv2d_fwd_xdl_dynamic_op_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_dynamic_op_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_dynamic_op_nhwgc_gkyxc_nhwgk_f32_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_dynamic_op_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

add_instance_library(device_grouped_conv2d_fwd_dynamic_op_nhwgc_instance ${GROUPED_CONV2D_FWD_DYNAMIC_OP_NHWGC})
