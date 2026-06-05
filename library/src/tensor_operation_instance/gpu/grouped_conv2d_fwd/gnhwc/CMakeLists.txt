# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNHWC layout instances for grouped_conv2d_fwd
set(GROUPED_CONV2D_FWD_GNHWC
   # dl
   dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   # wmma
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
   wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
   # xdl
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
)

add_instance_library(device_grouped_conv2d_fwd_gnhwc_instance ${GROUPED_CONV2D_FWD_GNHWC})
