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

struct AddReluAdd
{
    template <typename T1, typename T2>
    __host__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        float b = v0 + v1;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
    }

    template <typename T1, typename T2>
    __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
#if 0
        float a = v1 + v0;
        float b = max(a, float(0));
        float c = b + v2;

        return c;
#else
        float b = v1 + v2;
        float c = (v0 > -v1) ? b + v0 : v2;

        return c;
#endif
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
