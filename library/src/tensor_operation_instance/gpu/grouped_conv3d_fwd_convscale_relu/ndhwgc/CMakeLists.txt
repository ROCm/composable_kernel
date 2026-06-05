# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NDHWGC layout instances for grouped_conv3d_fwd_convscale_relu
set(GROUPED_CONV3D_FWD_CONVSCALE_RELU_NDHWGC
   # wmma
   wmma/device_grouped_conv3d_fwd_wmma_combconvscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instance.cpp
   wmma/device_grouped_conv3d_fwd_wmma_convscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_instance.cpp
   # xdl
   xdl/device_grouped_conv3d_fwd_xdl_combconvscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_convscale_relu_ndhwgc_gkzyxc_ndhwgk_f8_instance.cpp
)

add_instance_library(device_grouped_conv3d_fwd_convscale_relu_ndhwgc_instance ${GROUPED_CONV3D_FWD_CONVSCALE_RELU_NDHWGC})
