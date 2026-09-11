# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NWGC layout instances for grouped_conv1d_bwd_weight
set(GROUPED_CONV1D_BWD_WEIGHT_NWGC)

if(DL_KERNELS)
    list(APPEND GROUPED_CONV1D_BWD_WEIGHT_NWGC
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instance.cpp
        dl/device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instance.cpp)
endif()

add_instance_library(device_grouped_conv1d_bwd_weight_nwgc_instance ${GROUPED_CONV1D_BWD_WEIGHT_NWGC})
