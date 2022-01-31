#ifndef CK_ELEMENT_WISE_OPERATION_HPP
#define CK_ELEMENT_WISE_OPERATION_HPP

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        y = x;
    }
};

struct AddRelu
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const
    {
#if 1
        T a = x0 + x1;
        y   = a > 0 ? a : 0;
#else
        // hack for Hardswich
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        y       = c;
#endif
    }
};

struct AddReluAdd
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1, const T& x2) const
    {
#if 1
        T a = x0 + x1;
        T b = a > 0 ? a : 0;
        y   = b + x2;
#else
        // hack for Hardswich
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
#endif
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
