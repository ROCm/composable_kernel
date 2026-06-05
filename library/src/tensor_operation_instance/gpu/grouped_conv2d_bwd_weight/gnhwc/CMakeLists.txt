# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNHWC layout instances for grouped_conv2d_bwd_weight
set(GROUPED_CONV2D_BWD_WEIGHT_GNHWC
   # xdl
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_default_pipev1_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_pad0_pipev1_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_default_pipev1_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_pad0_pipev1_instance.cpp
)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV2D_BWD_WEIGHT_GNHWC
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
        dl/device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp)
endif()

add_instance_library(device_grouped_conv2d_bwd_weight_gnhwc_instance ${GROUPED_CONV2D_BWD_WEIGHT_GNHWC})
