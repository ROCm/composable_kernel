// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::builder {

// Enumeration for CK Device Operation types.
// This allows the builder to select which device operation template to instantiate
// based on the user's requirements.
enum class DeviceOpType
{
    // Forward Convolution - Non-grouped
    CONV_FWD, // Maps to: DeviceConvFwd (TODO: No implementation with tuning params exists yet)

    // Forward Convolution - Grouped
    GROUPED_CONV_FWD_MULTIPLE_ABD, // Maps to: DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
    GROUPED_CONV_FWD_MULTIPLE_ABD_XDL_CSHUFFLE_V3, // Maps to:
                                                   // DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
};

} // namespace ck_tile::builder
