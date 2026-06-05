# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# grouped_conv3d_bwd_data - layout-specific instances

# GNDHWC - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(gndhwc)
endif()
# NDHWGC - Required by MIOpen
add_subdirectory(ndhwgc)
# NGCDHW - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(ngcdhw)
endif()
