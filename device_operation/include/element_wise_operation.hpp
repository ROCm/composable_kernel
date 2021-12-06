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

struct AddRelu
{
    template <typename T1>
    __host__ constexpr float operator()(float v0, T1 v1) const
    {
        float b = v0 + v1;
        float c = b > 0 ? b : 0;

        return c;
    }

    template <typename T1>
    __device__ constexpr float operator()(float v0, T1 v1) const
    {
#if 0
        float a = v1 + v0;
        float b = max(a, float(0));

        return b;
#else
        float b = v1 + v0;
        float c = b > 0 ? b : 0;

        return c;
#endif
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

struct AddLeakyReluAdd
{
    template <typename T1, typename T2>
    __host__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        float a = v0 + v1;
        float b = 0.1 * a;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
    }

    template <typename T1, typename T2>
    __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
#if 0
        // this use not too many registers, but use fp64 mul
        float a = v0 + v1;
        float b = 0.1 * a;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
#elif 0
        // this spill register
        float a = v0 + v1;
        float b = float(0.1) * a;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
#elif 0
        // this use lots of registers (but no spill)
        constexpr float alpha     = 0.1;
        constexpr float alpha_inv = 1.0 / alpha;

        float a = v2 * alpha_inv;
        float b = v1 + v0;
        float c = b > 0 ? b : 0;
        float d = alpha * (a + c);

        return d;
#elif 1
        // this use lots of registers (but no spill), 89 Tflops
        constexpr float alpha     = 0.1;
        constexpr float alpha_inv = 1.0 / alpha;

        float a = v2 * alpha_inv;
        float b = v1 + v0;
        float c = max(b, float(0));
        float d = alpha * (a + c);

        return d;
#elif 1
        // this spill registers, 89 Tflops
        float a     = v0 + v1;
        float alpha = 0.1;

        float b;
        asm volatile("\n \
                v_mul_f32_e32 %0, %1, %2 \n \
                "
                     : "=v"(b)
                     : "s"(alpha), "v"(a));

        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
#endif
    }
};
} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
