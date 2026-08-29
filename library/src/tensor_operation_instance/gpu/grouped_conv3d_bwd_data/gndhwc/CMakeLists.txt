# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GNDHWC layout instances for grouped_conv3d_bwd_data
set(GROUPED_CONV3D_BWD_DATA_GNDHWC
   # wmma
   wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instance.cpp
   wmma/device_grouped_conv3d_bwd_data_wmma_gndhwc_gkzyxc_gndhwk_i8_instance.cpp
   # xdl
   xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_bf16_instance.cpp
   xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f16_instance.cpp
   xdl/device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f32_instance.cpp
)

add_instance_library(device_grouped_conv3d_bwd_data_gndhwc_instance ${GROUPED_CONV3D_BWD_DATA_GNDHWC})
