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

} // namespace device
} // namespace tensor_operation
} // namespace ck
