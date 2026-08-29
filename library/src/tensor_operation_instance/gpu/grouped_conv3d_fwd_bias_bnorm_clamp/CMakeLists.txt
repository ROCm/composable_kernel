# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# grouped_conv3d_fwd_bias_bnorm_clamp - layout-specific instances

# NDHWGC - Required by MIOpen
add_subdirectory(ndhwgc)
# NHWGC - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(nhwgc)
endif()
