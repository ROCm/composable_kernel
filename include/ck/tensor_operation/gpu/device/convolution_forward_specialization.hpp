#ifndef CONVOLUTION_FORWARD_SPECIALIZATION
#define CONVOLUTION_FORWARD_SPECIALIZATION

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionForwardSpecialization_t
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    OddC,
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
