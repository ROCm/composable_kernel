# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNWC layout instances for grouped_conv1d_fwd
set(GROUPED_CONV1D_FWD_GNWC
   xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_bf16_instance.cpp
   xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f16_instance.cpp
   xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f32_instance.cpp
   xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_int8_instance.cpp
)

add_instance_library(device_grouped_conv1d_fwd_gnwc_instance ${GROUPED_CONV1D_FWD_GNWC})
