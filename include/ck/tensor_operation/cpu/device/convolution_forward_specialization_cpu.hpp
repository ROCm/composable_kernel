#ifndef CONVOLUTION_FORWARD_SPECIALIZATION_CPU
#define CONVOLUTION_FORWARD_SPECIALIZATION_CPU

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {

enum ConvolutionForwardSpecialization_t
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    OddC,
};

} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
#endif
