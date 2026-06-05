# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# grouped_conv2d_fwd - layout-specific instances

# GNHWC - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(gnhwc)
endif()
# NGCHW - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(ngchw)
endif()
# NHWGC - Required by MIOpen
add_subdirectory(nhwgc)
