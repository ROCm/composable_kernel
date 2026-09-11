# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NHWGC layout instances for grouped_conv3d_fwd_bias_bnorm_clamp
set(GROUPED_CONV3D_FWD_BIAS_BNORM_CLAMP_NHWGC
   wmma/device_grouped_conv3d_fwd_bias_bn_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   wmma/device_grouped_conv3d_fwd_bias_bn_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance.cpp
)

add_instance_library(device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance ${GROUPED_CONV3D_FWD_BIAS_BNORM_CLAMP_NHWGC})
