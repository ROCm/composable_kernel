#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"


template <typename T>
constexpr bool always_false = false;

template <typename Y, typename X>
inline __host__ __device__ Y fast_type_convert(X x)
{
    static_assert(always_false<X>, "not implemented");
    (void)x;
}

template <>
inline __host__ __device__ ck::half_t fast_type_convert<ck::half_t, ck::f8_t>(ck::f8_t x)
{
    constexpr const uint16_t mask      = 0x7fff;
    constexpr const uint16_t sign_mask = 0x8000;
    // constexpr const uint16_t exp_compensate = 0x2000;  // for float8_e4m3fn
    constexpr const uint16_t exp_compensate = 0x1c00; // for float8_e4m3fnuz

    uint8_t x_u8   = reinterpret_cast<uint8_t&>(x);
    uint16_t x_u16 = static_cast<uint16_t>(x_u8) << 8;
    uint16_t exp   = (x_u16 & mask) >> 1;
    uint16_t y     = (x_u16 & sign_mask) | (exp + exp_compensate);
    return reinterpret_cast<ck::half_t&>(y);
}

template <>
inline __host__ __device__ ck::half2_t
fast_type_convert<ck::half2_t, ck::packed_f4x2_t>(ck::packed_f4x2_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
    uint8_t x_l  = (x_u8 & 0x0f) >> 0;
    uint8_t x_h  = (x_u8 & 0xf0) >> 4;

    uint8_t l_s  = x_l & 0x8;
    uint8_t l_em = x_l & 0x7;
    uint8_t l_u8 = (l_s << 4) | (l_em << 2);
    l_u8 += 0x38;
    // l_u8 = 0;

    uint8_t h_s  = x_h & 0x8;
    uint8_t h_em = x_h & 0x7;
    uint8_t h_u8 = (h_s << 4) | (h_em << 2);
    h_u8 += 0x38;
    // h_u8= 0;

    auto l_f16 = ck::type_convert<ck::half_t>(ck::bit_cast<ck::f8_t>(l_u8));
    auto h_f16 = ck::type_convert<ck::half_t>(ck::bit_cast<ck::f8_t>(h_u8));

    return {l_f16, h_f16};
}


struct ElementwiseScale
{
    __host__ __device__ void
    operator()(ck::half2_t& y, const ck::packed_f4x2_t& x0, const ck::half2_t& x1) const
    {
        auto scale = fast_type_convert<ck::half2_t>(x0);
        y          = scale * x1;
    }

    constexpr const static bool is_pack2_invocable = true;
};


int main() {
  ck::packed_f4x2_t v = 0;
  auto result = ck::bit_cast<half2>(fast_type_convert<ck::half2_t>(v));
  // auto result = ck::type_convert<ck::float2_t>(tmp);
  std::cout << ck::type_convert<float>(result.data[0]) << "\n";
  std::cout << ck::type_convert<float>(result.data[1]) << "\n";
}
