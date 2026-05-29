// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/joinable_thread.hpp"
#include "ck_tile/core/numeric/e4m3.hpp"
#include "ck_tile/core/numeric/e5m3.hpp"
#include "ck_tile/core/numeric/e8m0.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <unordered_set>
#include <vector>

using namespace ck_tile;

namespace test {

// Test fixture for FillUniformDistribution tests
template <typename T>
class FillUniformDistributionTest : public ::testing::Test
{
    public:
    static constexpr uint32_t seed = 42;
    static constexpr float a       = -5.0f;
    static constexpr float b       = 5.0f;
};

using TestTypes = ::testing::Types<float, fp16_t, fp8_t, pk_fp4_t>;
TYPED_TEST_SUITE(FillUniformDistributionTest, TestTypes);

// Test that multiple runs with the same seed produce identical results
#ifndef _WIN32
TYPED_TEST(FillUniformDistributionTest, ConsistencyWithSameSeed)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    constexpr size_t size = 1024 * 1024 * 1024 / sizeof(T); // 1G

    std::vector<T> vec1(size);
    auto start = std::chrono::high_resolution_clock::now();
    FillUniformDistribution<T>{a, b, seed}(vec1.begin(), vec1.end());
    auto end   = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    std::cout << "Taking " << sec << " sec to fill 1GB of data of type " << typeid(T).name()
              << std::endl;

    const auto cpu_cores = max(32U, get_available_cpu_cores());
    for(auto num_threads_diff : {-3, -1})
    {
        cpu_core_guard cg(min(max(cpu_cores + num_threads_diff, 1U), get_available_cpu_cores()));
        std::vector<T> vec2(size);
        FillUniformDistribution<T>{a, b, seed}(vec2.begin(), vec2.end());
        EXPECT_EQ(0, std::memcmp(vec1.data(), vec2.data(), size * sizeof(T)))
            << "First and second fill should be identical";
    }
}
#endif

// Test consistency across different data sizes (which affects threading)
TYPED_TEST(FillUniformDistributionTest, ConsistencyAcrossSizes)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    std::vector<size_t> test_sizes = {
        100,     // Small - likely single threaded
        10000,   // Medium
        1000000, // Large - will use multiple threads
        5000000  // Very large - will use many threads
    };

    for(size_t size : test_sizes)
    {
        std::vector<T> reference(size);
        std::vector<T> test_vec(size);

        FillUniformDistribution<T>{a, b, seed}(reference.begin(), reference.end());

        // Run multiple times to ensure consistency
        for(int run = 0; run < 3; ++run)
        {
            std::fill(test_vec.begin(), test_vec.end(), T{});
            FillUniformDistribution<T>{a, b, seed}(test_vec.begin(), test_vec.end());

            EXPECT_EQ(0, std::memcmp(reference.data(), test_vec.data(), size * sizeof(T)))
                << "Mismatch for size=" << size << " run=" << run;
        }
    }
}

// Test that different seeds produce different results
TYPED_TEST(FillUniformDistributionTest, CommonPrefix)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    std::vector<size_t> test_sizes = {
        100,     // Small - likely single threaded
        10000,   // Medium
        1000000, // Large - will use multiple threads
        5000000  // Very large - will use many threads
    };

    auto longest = std::make_unique<std::vector<T>>(test_sizes[0]);
    FillUniformDistribution<T>{a, b, seed}(longest->begin(), longest->end());
    for(size_t i = 1; i < test_sizes.size(); ++i)
    {
        auto current = std::make_unique<std::vector<T>>(test_sizes[i]);
        FillUniformDistribution<T>{a, b, seed}(current->begin(), current->end());
        size_t min_size = std::min(longest->size(), current->size());
        EXPECT_EQ(0, std::memcmp(longest->data(), current->data(), min_size * sizeof(T)))
            << "Different sizes with same seed should have the same prefix";
        if(current->size() > longest->size())
        {
            longest = std::move(current);
        }
    }
}

// Test edge cases
TYPED_TEST(FillUniformDistributionTest, EdgeCases)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    // Empty range
    std::vector<T> empty_vec;
    EXPECT_NO_THROW((FillUniformDistribution<T>{a, b, seed}(empty_vec.begin(), empty_vec.end())));

    // Single element
    std::vector<T> single1(1);
    std::vector<T> single2(1);
    FillUniformDistribution<T>{a, b, seed}(single1.begin(), single1.end());
    FillUniformDistribution<T>{a, b, seed}(single2.begin(), single2.end());

    EXPECT_EQ(0, std::memcmp(single1.data(), single2.data(), sizeof(T)))
        << "Single element should be consistent";

    // Small sizes that might affect threading decisions
    std::vector<size_t> small_sizes = {2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    for(size_t size : small_sizes)
    {
        std::vector<T> vec1(size);
        std::vector<T> vec2(size);
        FillUniformDistribution<T>{a, b, seed}(vec1.begin(), vec1.end());
        FillUniformDistribution<T>{a, b, seed}(vec2.begin(), vec2.end());

        EXPECT_EQ(0, std::memcmp(vec1.data(), vec2.data(), size * sizeof(T)))
            << "Edge case failed for size=" << size;
    }
}
} // namespace test

// ============================================================
// FillUniformScaleDistribution tests
// ============================================================

namespace test_scale {

// Returns true if f is a finite, non-NaN float.
bool is_valid_float(float f) { return std::isfinite(f); }

// Returns true if f is an exact power of two (positive).
bool is_power_of_two(float f)
{
    if(f <= 0.f || !is_valid_float(f))
        return false;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return (bits & 0x007fffffu) == 0u; // mantissa bits all zero
}

// Compute the expected raw range for a given ScaleType and float bounds.
template <typename ScaleType>
static std::pair<int, int> expected_raw_range(float min_f, float max_f)
{
    constexpr int float_bias = 127;
    constexpr int type_bias  = ck_tile::numeric_traits<ScaleType>::bias;
    constexpr int mant_bits  = ck_tile::numeric_traits<ScaleType>::mant;
    const int ieee_min       = static_cast<int>(ck_tile::numeric_utils<float>::get_exponent(min_f));
    const int ieee_max       = static_cast<int>(ck_tile::numeric_utils<float>::get_exponent(max_f));
    // raw=0 excluded: decodes to 0.0 for e4m3/e5m3 and to 2^-127 for e8m0 - same
    // assumption as the implementation in FillUniformScaleDistribution.
    constexpr int raw_min = 1;
    constexpr int raw_max = static_cast<int>(ck_tile::numeric<ScaleType>::binary_max);
    const int scale       = 1 << mant_bits;
    const int min_r       = std::max(((ieee_min - float_bias) + type_bias) * scale, raw_min);
    const int max_r       = std::min(((ieee_max - float_bias) + type_bias) * scale, raw_max);
    return {min_r, max_r};
}

// ---- typed fixture -------------------------------------------------
template <typename ScaleType>
class FillUniformScaleDistributionTest : public ::testing::Test
{
};

using ScaleTypes = ::testing::Types<ck_tile::e8m0_t, ck_tile::e4m3_t, ck_tile::e5m3_t>;
TYPED_TEST_SUITE(FillUniformScaleDistributionTest, ScaleTypes);

// 1. No garbage: all generated values are finite (not NaN, not Inf).
TYPED_TEST(FillUniformScaleDistributionTest, NoGarbageValues)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{0.0625f, 4.0f, 42}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_TRUE(is_valid_float(f))
            << "NaN/Inf at index " << i
            << " raw=" << static_cast<int>(static_cast<typename S::type>(v));
        ++i;
    }
}

// 2. All generated raw bytes are within [min_r, max_r].
TYPED_TEST(FillUniformScaleDistributionTest, RawValuesInExpectedRange)
{
    using S                   = TypeParam;
    constexpr float min_scale = 0.0625f;
    constexpr float max_scale = 4.0f;
    auto [min_r, max_r]       = expected_raw_range<S>(min_scale, max_scale);
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{min_scale, max_scale, 7}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        int raw = static_cast<int>(static_cast<typename S::type>(v));
        EXPECT_GE(raw, min_r) << "raw below min at index " << i;
        EXPECT_LE(raw, max_r) << "raw above max at index " << i;
        ++i;
    }
}

// 3. Reproducibility: identical seed -> identical output.
TYPED_TEST(FillUniformScaleDistributionTest, SameSeedSameOutput)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> a({1000}), b({1000});
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, 99}(a.begin(), a.end());
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, 99}(b.begin(), b.end());
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(S)));
}

// 4. Different seeds produce different outputs (with overwhelming probability).
TYPED_TEST(FillUniformScaleDistributionTest, DifferentSeedsDifferentOutput)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> a({1000}), b({1000});
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, 1}(a.begin(), a.end());
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, 2}(b.begin(), b.end());
    EXPECT_NE(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(S)));
}

// 5. Single-value range: [v, v] -> all generated raw bytes fall in that exponent band.
TYPED_TEST(FillUniformScaleDistributionTest, SingleValueRange)
{
    using S               = TypeParam;
    constexpr float pivot = 1.0f;
    auto [min_r, max_r]   = expected_raw_range<S>(pivot, pivot);
    ck_tile::HostTensor<S> buf({2000});
    ck_tile::FillUniformScaleDistribution<S>{pivot, pivot, 5}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        int raw = static_cast<int>(static_cast<typename S::type>(v));
        EXPECT_GE(raw, min_r) << "index " << i;
        EXPECT_LE(raw, max_r) << "index " << i;
        EXPECT_TRUE(is_valid_float(static_cast<float>(v))) << "index " << i;
        ++i;
    }
}

// 6. Non-power-of-two bounds snap to the nearest lower power-of-two exponent.
//    Verify outputs are still within the snapped raw range.
TYPED_TEST(FillUniformScaleDistributionTest, NonPowerOfTwoBoundsSnap)
{
    using S = TypeParam;
    // 0.1 snaps to 0.0625 (2^-4); 3.5 snaps to 2.0 (2^1)
    auto [min_r, max_r] = expected_raw_range<S>(0.0625f, 2.0f); // snapped bounds
    ck_tile::HostTensor<S> buf({5000});
    ck_tile::FillUniformScaleDistribution<S>{0.1f, 3.5f, 13}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        int raw = static_cast<int>(static_cast<typename S::type>(v));
        EXPECT_GE(raw, min_r) << "index " << i;
        EXPECT_LE(raw, max_r) << "index " << i;
        ++i;
    }
}

// 7. Coverage: for a small range, every possible raw value appears at least once
//    after enough samples (probabilistic - extremely unlikely to fail with 100k draws).
TYPED_TEST(FillUniformScaleDistributionTest, AllRawValuesGenerated)
{
    using S                 = TypeParam;
    constexpr float min_f   = 0.5f;
    constexpr float max_f   = 2.0f;
    auto [min_r, max_r]     = expected_raw_range<S>(min_f, max_f);
    const int range_size    = max_r - min_r + 1;
    const std::size_t draws = static_cast<std::size_t>(range_size) * 5000;

    ck_tile::HostTensor<S> buf({draws});
    ck_tile::FillUniformScaleDistribution<S>{min_f, max_f, 77}(buf.begin(), buf.end());

    std::unordered_set<int> seen;
    for(auto& v : buf)
        seen.insert(static_cast<int>(static_cast<typename S::type>(v)));

    EXPECT_EQ(static_cast<int>(seen.size()), range_size)
        << "Expected " << range_size << " distinct raw values, got " << seen.size();
}

// 8. e8m0 specific: all generated values must be exact powers of two.
TEST(FillUniformScaleDistributionE8M0, AllValuesPowersOfTwo)
{
    using S = ck_tile::e8m0_t;
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{0.0625f, 4.0f, 33}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_TRUE(is_power_of_two(f)) << "Non-power-of-two at index " << i << " value=" << f;
        ++i;
    }
}

// 9. Wide range stress: large tensor, wide float range, no garbage.
TYPED_TEST(FillUniformScaleDistributionTest, WideRangeStress)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> buf({50000});
    ck_tile::FillUniformScaleDistribution<S>{1.f / 1024, 1024.f, 0}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_TRUE(is_valid_float(f)) << "Bad value at index " << i;
        EXPECT_GT(f, 0.f) << "Non-positive scale at index " << i;
        ++i;
    }
}

// 10. Empty range does not crash.
TYPED_TEST(FillUniformScaleDistributionTest, EmptyRangeNoCrash)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> buf({0});
    EXPECT_NO_THROW(
        (ck_tile::FillUniformScaleDistribution<S>{1.0f, 1.0f, 0}(buf.begin(), buf.end())));
}

// 11. For e8m0 (mant=0), every generated value is exactly within [min_scale, max_scale].
//     Each exponent band has exactly one value so no overshoot is possible.
TEST(FillUniformScaleDistributionE8M0, StrictFloatBounds)
{
    using S               = ck_tile::e8m0_t;
    constexpr float min_f = 0.0625f, max_f = 4.0f;
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{min_f, max_f, 11}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_GE(f, min_f) << "index " << i;
        EXPECT_LE(f, max_f) << "index " << i;
        ++i;
    }
}

// 12. Unlike test 11 (e8m0 only, both bounds strict), this test covers all ExMy types and
//     checks only the upper bound. The lower bound is not strict for types with non-zero
//     mantissa bits (e4m3/e5m3): mantissa bits allow values between consecutive
//     power-of-two exponents, so some generated values can fall below min_scale (test 13
//     verifies this). The upper bound IS strict for all types because max_r is set to the
//     exact power-of-two raw encoding (mant=0), so the highest output is exactly max_scale_.
TYPED_TEST(FillUniformScaleDistributionTest, StrictFloatUpperBound)
{
    using S               = TypeParam;
    constexpr float min_f = 0.0625f, max_f = 4.0f;
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{min_f, max_f, 22}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_LE(f, max_f) << "value " << f << " exceeds max_scale at index " << i;
        ++i;
    }
}

// 13. When min_scale is not an exact power of two it snaps down to the nearest lower
//     power-of-two exponent, so some generated values will be below min_scale.
TYPED_TEST(FillUniformScaleDistributionTest, NonPowerOfTwoMinSnapsBelow)
{
    using S = TypeParam;
    // 0.1 is not a power of two; get_exponent snaps it down to 0.0625 (2^-4).
    // Values in [0.0625, 0.1) are therefore reachable.
    constexpr float min_f = 0.1f;
    ck_tile::HostTensor<S> buf({10000});
    ck_tile::FillUniformScaleDistribution<S>{min_f, 4.0f, 33}(buf.begin(), buf.end());
    bool found_below = false;
    for(auto& v : buf)
        if(static_cast<float>(v) < min_f)
            found_below = true;
    EXPECT_TRUE(found_below)
        << "Expected some values below non-power-of-two min_scale due to exponent snapping";
}

// 14. Extreme bounds that exceed the type's representable range clamp safely
//     and still produce only finite, positive values - no NaN, no crash.
TYPED_TEST(FillUniformScaleDistributionTest, ExtremeOutOfRangeBoundsClampSafely)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> buf({5000});
    ck_tile::FillUniformScaleDistribution<S>{1e-38f, 1e38f, 55}(buf.begin(), buf.end());
    std::size_t i = 0;
    for(const S& v : buf)
    {
        float f = static_cast<float>(v);
        EXPECT_TRUE(std::isfinite(f)) << "index " << i;
        EXPECT_GT(f, 0.f) << "index " << i;
        ++i;
    }
}

// 15. nullopt seed: two calls produce different outputs (random device seeding).
TYPED_TEST(FillUniformScaleDistributionTest, NulloptSeedProducesRandomOutput)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> a({500}), b({500});
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, std::nullopt}(a.begin(), a.end());
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, std::nullopt}(b.begin(), b.end());
    EXPECT_NE(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(S)));
}

// 16. Range overload: passing a ck_tile::HostTensor directly compiles and fills correctly.
TYPED_TEST(FillUniformScaleDistributionTest, RangeOverloadFillsHostTensor)
{
    using S = TypeParam;
    ck_tile::HostTensor<S> buf({1000});
    ck_tile::FillUniformScaleDistribution<S>{0.125f, 2.0f, 7}(buf);
    auto [min_r, max_r] = expected_raw_range<S>(0.125f, 2.0f);
    std::size_t i       = 0;
    for(const S& v : buf)
    {
        int raw = static_cast<int>(static_cast<typename S::type>(v));
        EXPECT_GE(raw, min_r) << "index " << i;
        EXPECT_LE(raw, max_r) << "index " << i;
        ++i;
    }
}

} // namespace test_scale
