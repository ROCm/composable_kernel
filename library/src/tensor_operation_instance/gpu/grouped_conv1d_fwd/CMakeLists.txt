# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# grouped_conv1d_fwd - layout-specific instances

# GNWC - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(gnwc)
endif()
