# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NDHWGC layout instances for grouped_conv3d_fwd_convinvscale
set(GROUPED_CONV3D_FWD_CONVINVSCALE_NDHWGC
   # wmma
   wmma/device_grouped_conv3d_fwd_wmma_convinvscale_ndhwgc_gkzyxc_ndhwgk_f8_instance.cpp
   # xdl
   xdl/device_grouped_conv3d_fwd_xdl_convinvscale_ndhwgc_gkzyxc_ndhwgk_f8_instance.cpp
)

add_instance_library(device_grouped_conv3d_fwd_convinvscale_ndhwgc_instance ${GROUPED_CONV3D_FWD_CONVINVSCALE_NDHWGC})
