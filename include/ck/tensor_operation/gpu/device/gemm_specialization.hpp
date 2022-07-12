// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct GemmSpecialization
{
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
};

inline std::string getGemmSpecializationString(const GemmSpecialization& s)
{
    switch(s)
    {
    case GemmSpecialization::Default: return "Default";
    case GemmSpecialization::MPadding: return "MPadding";
    case GemmSpecialization::NPadding: return "NPadding";
    case GemmSpecialization::KPadding: return "KPadding";
    case GemmSpecialization::MNPadding: return "MNPadding";
    case GemmSpecialization::MKPadding: return "MKPadding";
    case GemmSpecialization::NKPadding: return "NKPadding";
    case GemmSpecialization::MNKPadding: return "MNKPadding";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
