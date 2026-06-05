# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNDHWC layout instances for grouped_conv3d_bwd_weight
set(GROUPED_CONV3D_BWD_WEIGHT_GNDHWC
   # xdl
   xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instance.cpp
   xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   xdl/gndhwc_gkzyxc_gndhwk/device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT_GNDHWC
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
        dl/device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp)
endif()

add_instance_library(device_grouped_conv3d_bwd_weight_gndhwc_instance ${GROUPED_CONV3D_BWD_WEIGHT_GNDHWC})
