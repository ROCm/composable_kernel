#ifndef ELEMENT_WISE_OPERATION_HPP
#define ELEMENT_WISE_OPERATION_HPP

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename T>
    __host__ __device__ constexpr T operator()(T v) const
    {
        return v;
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
