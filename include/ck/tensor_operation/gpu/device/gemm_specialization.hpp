#ifndef GEMM_SPECIALIZATION
#define GEMM_SPECIALIZATION

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

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
