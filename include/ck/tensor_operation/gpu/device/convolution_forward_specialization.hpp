// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef CK_CODE_GEN_RTC
#include <string>
#endif

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionForwardSpecialization
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    OddC,
    Filter3x3,
};

#ifndef CK_CODE_GEN_RTC
inline std::string getConvForwardSpecializationString(const ConvolutionForwardSpecialization& s)
{
    switch(s)
    {
    case ConvolutionForwardSpecialization::Default: return "Default";
    case ConvolutionForwardSpecialization::Filter1x1Pad0: return "Filter1x1Pad0";
    case ConvolutionForwardSpecialization::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case ConvolutionForwardSpecialization::OddC: return "OddC";
    case ConvolutionForwardSpecialization::Filter3x3: return "Filter3x3";
    default: return "Unrecognized specialization!";
    }
}
#endif

} // namespace device
} // namespace tensor_operation
} // namespace ck
