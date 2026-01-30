// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
    Filter3x3Stride1Pad1Dilation1 //_200x200_32_4x4 // Image 200x200, K=C=4, and 32 batches
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
    case ConvolutionForwardSpecialization::Filter3x3Stride1Pad1Dilation1: return "Filter3x3Stride1Pad1Dilation1";
    default: return "Unrecognized specialization!";
    }
}
#endif

} // namespace device
} // namespace tensor_operation
} // namespace ck
