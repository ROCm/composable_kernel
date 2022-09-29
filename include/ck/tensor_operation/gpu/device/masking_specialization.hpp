// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct MaskingSpecialization
{
    MaskDisabled,
    MaskOutUpperTriangle
};

inline std::string getMaskingSpecializationString(const MaskingSpecialization& s)
{
    switch(s)
    {
    case MaskingSpecialization::MaskDisabled: return "MaskDisabled";
    case MaskingSpecialization::MaskOutUpperTriangle: return "MaskOutUpperTriangle";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
