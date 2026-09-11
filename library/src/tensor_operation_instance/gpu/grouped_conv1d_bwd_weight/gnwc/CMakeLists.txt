# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNWC layout instances for grouped_conv1d_bwd_weight
set(GROUPED_CONV1D_BWD_WEIGHT_GNWC
   # xdl
   xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_f32_bf16_instance.cpp
   xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instance.cpp
   xdl/device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instance.cpp
)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV1D_BWD_WEIGHT_GNWC
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instance.cpp)
endif()

add_instance_library(device_grouped_conv1d_bwd_weight_gnwc_instance ${GROUPED_CONV1D_BWD_WEIGHT_GNWC})
