// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include <cmath>
#include <hip/hip_runtime.h>

#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using ::ck::hip_check_error;

using ck::bhalf_t;
using ck::bit_cast;
using ck::DeviceMem;
using ck::half_t;
using ck::type_convert;

// --- NumericLimits aliases (binary bit patterns) ---
const uint16_t bf16_inf     = bit_cast<uint16_t>(ck::NumericLimits<bhalf_t>::Infinity());
const uint16_t bf16_inf_neg = bf16_inf | 0x8000u;
const uint16_t bf16_qnan    = bit_cast<uint16_t>(ck::NumericLimits<bhalf_t>::QuietNaN());
const uint16_t bf16_max     = bit_cast<uint16_t>(ck::NumericLimits<bhalf_t>::Max());
const uint16_t bf16_low     = bit_cast<uint16_t>(ck::NumericLimits<bhalf_t>::Lowest());
const uint16_t bf16_minnorm = bit_cast<uint16_t>(ck::NumericLimits<bhalf_t>::Min());

const uint32_t f32_inf     = bit_cast<uint32_t>(ck::NumericLimits<float>::Infinity());
const uint32_t f32_inf_neg = f32_inf | 0x80000000u;
const uint32_t f32_qnan    = bit_cast<uint32_t>(ck::NumericLimits<float>::QuietNaN());
const uint32_t f32_max     = bit_cast<uint32_t>(ck::NumericLimits<float>::Max());
const uint32_t f32_minnorm = bit_cast<uint32_t>(ck::NumericLimits<float>::Min());

const uint16_t f16_inf     = bit_cast<uint16_t>(ck::NumericLimits<half_t>::Infinity());
const uint16_t f16_inf_neg = f16_inf | 0x8000u;
const uint16_t f16_qnan    = bit_cast<uint16_t>(ck::NumericLimits<half_t>::QuietNaN());
const uint16_t f16_max     = bit_cast<uint16_t>(ck::NumericLimits<half_t>::Max());
const uint16_t f16_minnorm = bit_cast<uint16_t>(ck::NumericLimits<half_t>::Min());

constexpr int8_t i8_max = ck::NumericLimits<int8_t>::Max();
constexpr int8_t i8_min = ck::NumericLimits<int8_t>::Min();

// --- Result map ---
template <typename T>
struct Bf16TestCase
{
    uint16_t bf16_bits;
    T T_bits;
    bool is_nan           = false;
    bool is_roundtrip_inf = false;
};

// --- Kernels ---
template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}

template <typename T>
struct CVT_FROM_BF16
{
    __host__ __device__ void operator()(const bhalf_t* src, T* dst, const int N) const
    {
        for(int i = 0; i < N; i++)
            dst[i] = type_convert<T>(src[i]);
    }
};

template <typename T>
struct CastRoundTrip
{
    __host__ __device__ void
    operator()(const T* src, bhalf_t* dst_bf16, T* dst_T, const int N) const
    {
        for(int i = 0; i < N; i++)
        {
            dst_bf16[i] = type_convert<bhalf_t>(src[i]);
            dst_T[i]    = type_convert<T>(dst_bf16[i]);
        }
    }
};

// --- Helper: test bf16 -> T ---
template <typename OutT, typename BitT>
void test_from_bf16(const std::vector<Bf16TestCase<BitT>>& cases, bool on_device)
{
    int test_size = cases.size();

    std::vector<bhalf_t> inputs(test_size);
    for(int i = 0; i < test_size; i++)
        inputs[i] = bit_cast<bhalf_t>(cases[i].bf16_bits);

    std::vector<OutT> outputs(test_size);
    using job = CVT_FROM_BF16<OutT>;

    if(on_device)
    {
        DeviceMem device_in(test_size * sizeof(bhalf_t));
        DeviceMem device_out(test_size * sizeof(OutT));
        device_in.ToDevice(inputs.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const bhalf_t*>(device_in.GetDeviceBuffer()),
                                reinterpret_cast<OutT*>(device_out.GetDeviceBuffer()),
                                test_size);

        device_out.FromDevice(outputs.data());
    }
    else
    {
        job{}(inputs.data(), outputs.data(), test_size);
    }

    for(int i = 0; i < test_size; i++)
    {
        auto c = cases[i];
        if(c.is_nan)
        {
            EXPECT_TRUE(std::isnan(static_cast<float>(outputs[i])))
                << "NaN not preserved: bf16_bits=0x" << std::hex << c.bf16_bits << std::dec;
            continue;
        }

        EXPECT_EQ(c.T_bits, bit_cast<BitT>(outputs[i]))
            << " bf16=0x" << std::hex << c.bf16_bits << " expected=0x" << c.T_bits << " got=0x"
            << bit_cast<BitT>(outputs[i]) << std::dec;
    }
}

// --- Helper: test T -> bf16 -> T ---
template <typename T, typename BitT>
void test_roundtrip(const std::vector<Bf16TestCase<BitT>>& cases, bool on_device)
{
    int test_size = cases.size();

    std::vector<T> inputs(test_size);
    for(int i = 0; i < test_size; i++)
    {
        if constexpr(std::is_same_v<T, float>)
            inputs[i] = bit_cast<float>(cases[i].T_bits);
        else if constexpr(std::is_same_v<T, half_t>)
            inputs[i] = bit_cast<half_t>(cases[i].T_bits);
        else
            inputs[i] = cases[i].T_bits;
    }

    std::vector<bhalf_t> out_bf16(test_size);
    std::vector<T> outputs(test_size);
    using job = CastRoundTrip<T>;

    if(on_device)
    {
        DeviceMem device_in(test_size * sizeof(T));
        DeviceMem device_out_bf16(test_size * sizeof(bhalf_t));
        DeviceMem device_out(test_size * sizeof(T));

        device_in.ToDevice(inputs.data());
        MyKernel<job><<<1, 1>>>(reinterpret_cast<const T*>(device_in.GetDeviceBuffer()),
                                reinterpret_cast<bhalf_t*>(device_out_bf16.GetDeviceBuffer()),
                                reinterpret_cast<T*>(device_out.GetDeviceBuffer()),
                                test_size);

        device_out_bf16.FromDevice(out_bf16.data());
        device_out.FromDevice(outputs.data());
    }
    else
    {
        job{}(inputs.data(), out_bf16.data(), outputs.data(), test_size);
    }

    for(int i = 0; i < test_size; i++)
    {
        auto c = cases[i];
        // check T to bf16
        if(c.is_nan)
        {
            EXPECT_TRUE(std::isnan(type_convert<float>(out_bf16[i])))
                << "NaN not preserved: bf16=0x" << std::hex << bit_cast<uint16_t>(out_bf16[i])
                << std::dec;
            continue; // Skip when out_bf16 is inf (tested in : test_from_bf16)
        }
        EXPECT_EQ(c.bf16_bits, bit_cast<uint16_t>(out_bf16[i]))
            << "T -> bf16: expected=0x" << std::hex << c.bf16_bits << " got=0x"
            << bit_cast<uint16_t>(out_bf16[i]) << std::dec;

        // check Round Trip T to bf16 to T
        if(std::isinf(static_cast<float>(out_bf16[i])))
        {
            continue; // Skip when out_bf16 is inf (tested in : test_from_bf16)
        }
        else if(c.is_roundtrip_inf)
        {
            EXPECT_TRUE(std::isinf(static_cast<float>(outputs[i])))
                << "Expect Inf after round trip: got=0x" << std::hex << bit_cast<BitT>(outputs[i])
                << std::dec;
            continue;
        }
        else
        {
            const float tol = std::max(std::abs(static_cast<float>(inputs[i])) * std::pow(2.0f, -7),
                                       std::pow(2.0f, -7));
            ASSERT_NEAR(outputs[i], inputs[i], tol)
                << "Round trip: expected=0x" << std::hex << c.T_bits << " got=0x"
                << bit_cast<BitT>(outputs[i]) << std::dec;
        }
    }
}

// --- Host / Device tags for typed tests ---
struct Host
{
    static constexpr bool on_device = false;
};
struct Device
{
    static constexpr bool on_device = true;
};

struct RunModeNames
{
    template <typename T>
    static std::string GetName(int)
    {
        if constexpr(std::is_same_v<T, Host>)
            return "Host";
        else
            return "Device";
    }
};

template <typename T>
class BhalfConvertTest : public ::testing::Test
{
};

using RunModes = ::testing::Types<Host, Device>;
TYPED_TEST_SUITE(BhalfConvertTest, RunModes, RunModeNames);

TEST(BhalfTest, Traits)
{
    EXPECT_EQ(sizeof(bhalf_t), 2u);
    EXPECT_EQ(alignof(bhalf_t), alignof(uint16_t));
    EXPECT_TRUE(std::is_trivially_copyable_v<bhalf_t>);
}

TYPED_TEST(BhalfConvertTest, Bf16ToFloat)
{
    const std::vector<Bf16TestCase<uint32_t>> cases = {
        {0x0000u, 0x00000000u},      // +0
        {0x8000u, 0x80000000u},      // -0
        {bf16_inf, f32_inf},         // +inf             0x7F80
        {bf16_inf_neg, f32_inf_neg}, // -inf             0xFF80
        {bf16_max, 0x7F7F0000u},     // max              0x7F7Fu
        {bf16_low, 0xFF7F0000u},     // lowest           0xFF7Fu
        {bf16_minnorm, f32_minnorm}, // min normal       0x0080u
        {0x007Fu, 0x007F0000u},      // max subnormal
        {0x0001u, 0x00010000u},      // min subnormal
        {bf16_qnan, f32_qnan, true}, // NaN              0x7FC0
    };
    test_from_bf16<float>(cases, TypeParam::on_device);
}

TYPED_TEST(BhalfConvertTest, Bf16ToHalf)
{
    const std::vector<Bf16TestCase<uint16_t>> cases = {
        {0x0000u, 0x0000u},          // +0
        {0x8000u, 0x8000u},          // -0
        {bf16_inf, f16_inf},         // +inf             0x7F80
        {bf16_inf_neg, f16_inf_neg}, // -inf             0xFF80
        {bf16_max, f16_inf},         // max              0x7F7Fu
        {bf16_low, f16_inf_neg},     // lowest           0xFF7Fu
        {bf16_minnorm, 0x0000u},     // min normal       0x0080u
        {0x007Fu, 0x0000u},          // max subnormal
        {0x0001u, 0x0000u},          // min subnormal
        {bf16_qnan, f16_qnan, true}, // NaN              0x7FC0
    };
    test_from_bf16<half_t>(cases, TypeParam::on_device);
}

TYPED_TEST(BhalfConvertTest, Bf16ToInt8)
{
    const std::vector<Bf16TestCase<int8_t>> cases = {
        // truncation toward zero spanning int8 range
        {0x3F40u, 0},    // 0.75 -> 0
        {0xBF40u, 0},    // -0.75 -> 0
        {0x3FC0u, 1},    // 1.5 -> 1
        {0xC0F0u, -7},   // -7.5 -> -7
        {0x422Bu, 42},   // 42.75 -> 42
        {0xC27Eu, -63},  // -63.5 -> -63
        {0x42FDu, 126},  // 126.5 -> 126
        {0xC2FFu, -127}, // -127.5 -> -127.
    };
    test_from_bf16<int8_t>(cases, TypeParam::on_device);
}

TYPED_TEST(BhalfConvertTest, FloatRoundTrip)
{
    const std::vector<Bf16TestCase<uint32_t>> cases = {
        // special values
        {0x0000u, 0x00000000u},       // +0
        {0x8000u, 0x80000000u},       // -0
        {bf16_inf, f32_inf},          // +inf
        {bf16_inf_neg, f32_inf_neg},  // -inf
        {0xFFFFu, 0xFFFFFFFFu, true}, // NaN
        // boundary
        {bf16_inf, f32_max},         // float max -> bf16 +inf
        {0x0000u, 0x00000001u},      // float min subnorm -> bf16 0
        {bf16_minnorm, 0x007FFFFFu}, // float max subnorm -> bf16 min norm
        {bf16_minnorm, f32_minnorm}, // float min norm -> bf16 min norm
        {0x8200u, 0x81FFFFFFu},      // mantissa overflow
        // normal values spanning the range
        {0x2580u, bit_cast<uint32_t>(std::ldexp(1.0f, -52))}, // 2^-52
        {0x3200u, bit_cast<uint32_t>(std::ldexp(1.0f, -27))}, // 2^-27
        {0xBE00u, bit_cast<uint32_t>(-0.125f)},               // -0.125
        {0x3F80u, bit_cast<uint32_t>(1.0f)},                  // 1.0
        {0x4228u, bit_cast<uint32_t>(42.0f)},                 // 42.0
        {0xC348u, bit_cast<uint32_t>(-200.0f)},               // -200.0
        {0x4E80u, bit_cast<uint32_t>(std::ldexp(1.0f, 30))},  // 2^30
        {0x7F00u, bit_cast<uint32_t>(std::ldexp(1.0f, 127))}, // 2^127
        // RNE rounding near 1.0 (bf16 step = 2^-7)
        {0x3F80u, 0x3F808000u}, // tie, even LSB -> round down
        {0x3F82u, 0x3F818000u}, // tie, odd LSB -> round up
        {0x3F80u, 0x3F804000u}, // below tie -> round down
        {0x3F81u, 0x3F80C000u}, // above tie -> round up
        // RNE rounding near 256.0 (bf16 step = 2, float ULP = 2^-15)
        {0x4380u, 0x43808000u}, // 257: tie, even LSB -> round down to 256
        {0x4382u, 0x43818000u}, // 259: tie, odd LSB -> round up to 260
        {0x4380u, 0x43804000u}, // 256.5: below tie -> round down to 256
        {0x4381u, 0x4380C000u}, // 257.5: above tie -> round up to 258
    };
    test_roundtrip<float>(cases, TypeParam::on_device);
}

TYPED_TEST(BhalfConvertTest, F16RoundTrip)
{
    const std::vector<Bf16TestCase<uint16_t>> cases = {
        // special values
        {0x0000u, 0x0000u},          // +0
        {0x8000u, 0x8000u},          // -0
        {bf16_inf, f16_inf},         // +inf
        {bf16_inf_neg, f16_inf_neg}, // -inf
        {bf16_qnan, f16_qnan, true}, // NaN
        // boundary
        {0x3380u, 0x0001u},              // fp16 min subnorm (2^-24) -> bf16 normal
        {0x3880u, 0x03FFu},              // fp16 max subnorm -> rounds up to bf16 2^-14
        {0x3880u, f16_minnorm},          // fp16 min normal (2^-14) -> bf16 exact
        {0x4780u, f16_max, false, true}, // fp16 max (65504) -> bf16 65536 -> fp16 +inf
        // normal values spanning fp16 range, exact in both fp16 and bf16
        {0xBA80u, 0x9400u}, // -2^-10 ~= -9.77e-4
        {0x3C80u, 0x2400u}, // 2^-6 = 0.015625
        {0xC060u, 0xC300u}, // -3.5
        {0x3F80u, 0x3C00u}, // 1.0
        {0x4188u, 0x4C40u}, // 17.0
        {0xC348u, 0xDA40u}, // -200.0
        {0x4580u, 0x6C00u}, // 4096.0
        {0x4720u, 0x7900u}, // 40960.0
        // RNE rounding near 1.0 (bf16 step = 2^-7, fp16 ULP = 2^-10)
        {0x3F80u, 0x3C04u}, // tie, even LSB -> round down
        {0x3F82u, 0x3C0Cu}, // tie, odd LSB -> round up
        {0x3F80u, 0x3C02u}, // below tie -> round down
        {0x3F81u, 0x3C06u}, // above tie -> round up
        // RNE rounding near 1024 (bf16 step = 8, fp16 ULP = 1)
        {0x4480u, 0x6404u}, // 1028: tie, even LSB -> round down to 1024
        {0x4482u, 0x640Cu}, // 1036: tie, odd LSB -> round up to 1040
        {0x4480u, 0x6402u}, // 1026: below tie -> round down to 1024
        {0x4481u, 0x6406u}, // 1030: above tie -> round up to 1032
    };
    test_roundtrip<half_t>(cases, TypeParam::on_device);
}

TYPED_TEST(BhalfConvertTest, Int8RoundTrip)
{
    // All int8 values [-128, 127] are exactly representable in bf16,
    // so every round-trip is lossless. No rounding to test.
    const std::vector<Bf16TestCase<int8_t>> cases = {
        // boundary
        {0x0000u, 0},      // 0
        {0x3F80u, 1},      // 1 (smallest positive)
        {0xBF80u, -1},     // -1 (largest negative)
        {0x42FEu, i8_max}, // int8 max
        {0xC300u, i8_min}, // int8 min
        // powers of 2
        {0x4000u, 2},   // 2
        {0xC080u, -4},  // -4
        {0x4180u, 16},  // 16
        {0xC200u, -32}, // -32
        {0x4280u, 64},  // 64
        // spanning values
        {0x40E0u, 7},    // 7
        {0xC150u, -13},  // -13
        {0x4228u, 42},   // 42
        {0xC2C8u, -100}, // -100
        {0x42C6u, 99},   // 99
    };
    test_roundtrip<int8_t>(cases, TypeParam::on_device);
}

using ck::bhalf2_t;
using ck::bhalf4_t;
using ck::bhalf8_t;

TEST(BhalfVecTest, VecTraits)
{
    EXPECT_EQ(sizeof(bhalf2_t), 2 * sizeof(bhalf_t));
    EXPECT_EQ(sizeof(bhalf4_t), 4 * sizeof(bhalf_t));
    EXPECT_EQ(sizeof(bhalf8_t), 8 * sizeof(bhalf_t));
}

TEST(BhalfVecTest, Vec2ElementAccess)
{
    bhalf_t a = bit_cast<bhalf_t>(uint16_t{0x3F80}); // 1.0
    bhalf_t b = bit_cast<bhalf_t>(uint16_t{0x4228}); // 42.0

    bhalf2_t v = {a, b};

    EXPECT_EQ(bit_cast<uint16_t>(v[0]), 0x3F80u);
    EXPECT_EQ(bit_cast<uint16_t>(v[1]), 0x4228u);
}

TEST(BhalfVecTest, Vec4ElementAccess)
{
    const uint16_t patterns[] = {0x0000u, 0x3F80u, 0x4228u, 0xC348u};
    bhalf4_t v                = {bit_cast<bhalf_t>(patterns[0]),
                                 bit_cast<bhalf_t>(patterns[1]),
                                 bit_cast<bhalf_t>(patterns[2]),
                                 bit_cast<bhalf_t>(patterns[3])};

    for(int i = 0; i < 4; i++)
        EXPECT_EQ(bit_cast<uint16_t>(v[i]), patterns[i]) << "index=" << i;
}

TEST(BhalfVecTest, Vec8ElementAccess)
{
    const uint16_t patterns[] = {
        0x0000u, 0x8000u, 0x3F80u, 0xBF80u, 0x4228u, 0xC348u, bf16_inf, bf16_minnorm};
    bhalf8_t v{};
    for(int i = 0; i < 8; i++)
        v[i] = bit_cast<bhalf_t>(patterns[i]);

    for(int i = 0; i < 8; i++)
        EXPECT_EQ(bit_cast<uint16_t>(v[i]), patterns[i]) << "index=" << i;
}
