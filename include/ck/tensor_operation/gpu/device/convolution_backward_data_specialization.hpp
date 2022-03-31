#ifndef CONVOLUTION_BACKWARD_DATA_SPECIALIZATION
#define CONVOLUTION_BACKWARD_DATA_SPECIALIZATION

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionBackwardDataSpecialization
{
    Default,
    Filter1x1Stride1Pad0,
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
