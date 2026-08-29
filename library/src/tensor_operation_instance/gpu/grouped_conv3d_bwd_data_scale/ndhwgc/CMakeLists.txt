# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NDHWGC layout instances for grouped_conv3d_bwd_data_scale
set(GROUPED_CONV3D_BWD_DATA_SCALE_NDHWGC
   # wmma
   wmma/device_grouped_conv3d_bwd_data_wmma_v3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
   wmma/device_grouped_conv3d_bwd_data_wmma_v3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   # xdl
   xdl/device_grouped_conv3d_bwd_data_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
   xdl/device_grouped_conv3d_bwd_data_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   xdl/device_grouped_conv3d_bwd_data_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
   xdl/device_grouped_conv3d_bwd_data_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.cpp
)

add_instance_library(device_grouped_conv3d_bwd_data_scale_ndhwgc_instance ${GROUPED_CONV3D_BWD_DATA_SCALE_NDHWGC})
