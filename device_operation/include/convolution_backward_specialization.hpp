#ifndef CONVOLUTION_FORWARD_SPECIALIZATION
#define CONVOLUTION_FORWARD_SPECIALIZATION

namespace ck {
namespace tensor_operation {
namespace device {

enum ConvolutionBackwardSpecialization_t
{
    Default,
    Filter1x1Stride1Pad0,
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
