#ifndef CONVOLUTION_FORWARD_SPECIALIZATION
#define CONVOLUTION_FORWARD_SPECIALIZATION

#include <string>

namespace ck {
namespace tensor_operation {
namespace device {

enum ConvolutionForwardSpecialization_t
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    OddC,
};

inline std::string getConvFwdSpecializationStr(const ConvolutionForwardSpecialization_t& s)
{
    switch(s)
    {
    case Default: return "Default";
    case Filter1x1Pad0: return "Filter1x1Pad0";
    case Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case OddC: return "OddC";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
