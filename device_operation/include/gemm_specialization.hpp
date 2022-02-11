#ifndef GEMM_SPECIALIZATION
#define GEMM_SPECIALIZATION

namespace ck {
namespace tensor_operation {
namespace device {

enum GemmSpecialization_t
{
    NoPadding = 0,
    MNPadding,
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
