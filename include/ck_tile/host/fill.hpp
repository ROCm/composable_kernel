// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/host/joinable_thread.hpp"

namespace ck_tile {

/**
 * @brief Functor for filling a range with randomly generated values from a uniform distribution.
 *
 * This struct provides functionality to fill iterators or ranges with random values
 * generated from a uniform distribution. It supports both single-threaded and
 * multi-threaded operation.
 *
 * @tparam T The target type for the generated values.
 *
 * @note The multi-threaded implementation is not guaranteed to provide perfectly
 * distributed values across threads.
 *
 * @example
 *
 *     // Direct usage without creating a separate variable:
 *     ck_tile::FillUniformDistribution<>{-1.f, 1.f}(a_host_tensor);
 */
template <typename T = void>
struct FillUniformDistribution
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        if(first == last)
            return;
        using T_iter = std::decay_t<decltype(*first)>;
        static_assert(std::is_same_v<T, T_iter> || std::is_void_v<T>,
                      "Iterator value type must match template type T");
        constexpr auto PackedSize = numeric_traits<T_iter>::PackedSize;
        const auto total          = static_cast<size_t>(std::distance(first, last));
        const auto total_bytes    = total * sizeof(T_iter);

        // max 80 threads; at least 2MB per thread
        const size_t available_cpu_cores    = get_available_cpu_cores();
        constexpr uint64_t MAX_THREAD_COUNT = 80;
        const size_t num_thread             = min(
            MAX_THREAD_COUNT, available_cpu_cores, integer_divide_ceil(total_bytes, 0x200000UL));
        constexpr size_t BLOCK_BYTES   = 64;
        constexpr size_t BLOCK_SIZE    = BLOCK_BYTES / sizeof(T_iter);
        const size_t num_blocks        = integer_divide_ceil(total_bytes, BLOCK_BYTES);
        const size_t blocks_per_thread = integer_divide_ceil(num_blocks, num_thread);

        // use minstd_rand for better performance on discard()
        std::minstd_rand gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_real_distribution<float> dis(a_, b_);

        std::vector<joinable_thread> threads;
        threads.reserve(num_thread - 1); // last job run in the main thread
        for(int it = num_thread - 1; it >= 0; --it)
        {
            const size_t ib_begin = it * blocks_per_thread;
            const size_t ib_end   = min(ib_begin + blocks_per_thread, num_blocks);

            auto job = [=]() {
                auto g_ = gen; // copy
                auto d_ = dis; // copy
                g_.discard(ib_begin * BLOCK_SIZE * PackedSize);
                auto t_fn = [&]() {
                    if constexpr(PackedSize == 2)
                        return type_convert<T_iter>(fp32x2_t{d_(g_), d_(g_)});
                    else if constexpr(PackedSize == 16)
                    {
#if CK_TILE_AVX512F_WA
                        // Use fp32x8_t[2] workaround when AVX-512 is not supported
                        fp32x8_t tmp[2];
                        for(int i = 0; i < 8; ++i)
                        {
                            tmp[0][i] = d_(g_);
                            tmp[1][i] = d_(g_);
                        }
#else
                        fp32x16_t tmp{};
                        for(int i = 0; i < PackedSize; ++i)
                            tmp[i] = d_(g_);
#endif
                        return type_convert<T_iter>(tmp);
                    }
                    else
                        return type_convert<T_iter>(d_(g_));
                };

                size_t ib = ib_begin;
                for(; ib < ib_end - 1; ++ib) // full blocks
                    static_for<0, BLOCK_SIZE, 1>{}([&](auto iw_) {
                        constexpr size_t iw             = iw_.value;
                        *(first + ib * BLOCK_SIZE + iw) = t_fn();
                    });
                for(size_t iw = 0; iw < BLOCK_SIZE; ++iw) // last block
                    if(ib * BLOCK_SIZE + iw < total)
                        *(first + ib * BLOCK_SIZE + iw) = t_fn();
            };

            if(it > 0)
                threads.emplace_back(std::move(job));
            else
                job(); // last job run in the main thread
        }
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <>
struct FillUniformDistribution<ck_tile::pk_int4_t>
{
    float a_{-8.f}; // same type as primary template so that
                    // `FillUniformDistribution<Type>{-5.0f, 5.0f}` works for all types
    float b_{7.f};
    std::optional<uint32_t> seed_{11939};
    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        if(a_ < -8.0f || b_ > 7.0f)
        {
            throw std::runtime_error(
                "a_ or b_ of FillUniformDistribution<ck_tile::pk_int4_t> is out of range.");
        }

        int min_value             = static_cast<int>(a_);
        int max_value             = static_cast<int>(b_);
        constexpr auto int4_array = std::array<uint8_t, 16>{0x88,
                                                            0x99,
                                                            0xaa,
                                                            0xbb,
                                                            0xcc,
                                                            0xdd,
                                                            0xee,
                                                            0xff,
                                                            0x00,
                                                            0x11,
                                                            0x22,
                                                            0x33,
                                                            0x44,
                                                            0x55,
                                                            0x66,
                                                            0x77};
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_int_distribution<std::int32_t> dis(0, max_value - min_value + 1);
        while(first != last)
        {
            int randomInt = dis(gen);
            *first        = int4_array[randomInt + (min_value + 8)];
            ++first;
        }
    }
    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

namespace impl {

// clang-format off
template<index_t bytes> struct RawIntegerType_ {};
template<> struct RawIntegerType_<1> { using type = uint8_t;};
template<> struct RawIntegerType_<2> { using type = uint16_t;};
template<> struct RawIntegerType_<4> { using type = uint32_t;};
template<> struct RawIntegerType_<8> { using type = uint64_t;};
// clang-format on

template <typename T>
using RawIntegerType = typename RawIntegerType_<sizeof(T)>::type;
} // namespace impl

// Note: this struct will have no const-ness will generate random
template <typename T>
struct FillUniformDistribution_Unique
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    std::mt19937 gen_{};
    std::unordered_set<impl::RawIntegerType<T>> set_{};

    FillUniformDistribution_Unique(float a                      = -5.f,
                                   float b                      = 5.f,
                                   std::optional<uint32_t> seed = {11939})
        : a_(a),
          b_(b),
          seed_(seed),
          gen_{seed_.has_value() ? *seed_ : std::random_device{}()},
          set_{}
    {
    }

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last)
    {
        std::mt19937& gen = gen_;
        std::uniform_real_distribution<float> dis(a_, b_);
        auto& set = set_;
        std::generate(first, last, [&dis, &gen, &set]() {
            T v = static_cast<T>(0);
            do
            {
                v = ck_tile::type_convert<T>(dis(gen));
            } while(set.count(bit_cast<impl::RawIntegerType<T>>(v)) == 1);
            set.insert(bit_cast<impl::RawIntegerType<T>>(v));

            return v;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range)
        -> std::void_t<decltype(std::declval<FillUniformDistribution_Unique&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }

    void clear() { set_.clear(); }
};

template <typename T>
struct FillNormalDistribution
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};
    // ATTENTION: threaded does not guarantee the distribution between thread
    bool threaded = false;

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        if(threaded)
        {
            uint32_t num_thread  = std::thread::hardware_concurrency();
            auto total           = static_cast<std::size_t>(std::distance(first, last));
            auto work_per_thread = static_cast<std::size_t>((total + num_thread - 1) / num_thread);

            std::vector<joinable_thread> threads(num_thread);
            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end   = std::min((it + 1) * work_per_thread, total);
                auto thread_f        = [this, total, iw_begin, iw_end, &first] {
                    if(iw_begin > total || iw_end > total)
                        return;
                    // need to make each thread unique, add an offset to current seed
                    std::mt19937 gen(seed_.has_value() ? (*seed_ + iw_begin)
                                                       : std::random_device{}());
                    std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
                    std::generate(first + iw_begin, first + iw_end, [&dis, &gen]() {
                        return ck_tile::type_convert<T>(dis(gen));
                    });
                };
                threads[it] = joinable_thread(thread_f);
            }
        }
        else
        {
            std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
            std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
            std::generate(
                first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
        }
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

// Normally FillUniformDistributionIntegerValue should use std::uniform_int_distribution as below.
// However this produces segfaults in std::mt19937 which look like inifite loop.
//      template <typename T>
//      struct FillUniformDistributionIntegerValue
//      {
//          int a_{-5};
//          int b_{5};
//
//          template <typename ForwardIter>
//          void operator()(ForwardIter first, ForwardIter last) const
//          {
//              std::mt19937 gen(11939);
//              std::uniform_int_distribution<int> dis(a_, b_);
//              std::generate(
//                  first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
//          }
//      };

// Workaround for uniform_int_distribution not working as expected. See note above.<
template <typename T>
struct FillUniformDistributionIntegerValue
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillNormalDistributionIntegerValue
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillMonotonicSeq
{
    T init_value_{0};
    T step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, *this, n = init_value_]() mutable {
            auto tmp = n;
            if constexpr(std::is_same_v<decltype(tmp), pk_int4_t>)
            {
                n.data += step_.data;
            }
            else
            {
                n += step_;
            }
            return tmp;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillMonotonicSeq&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T, bool IsAscending = true>
struct FillStepRange
{
    float start_value_{0};
    float end_value_{3};
    float step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, *this, n = start_value_]() mutable {
            auto tmp = n;
            n += step_;
            if constexpr(IsAscending)
            {
                if(n > end_value_)
                    n = start_value_;
            }
            else
            {
                if(n < end_value_)
                    n = start_value_;
            }

            return type_convert<T>(tmp);
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillStepRange&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillConstant
{
    T value_{0};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::fill(first, last, value_);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillConstant&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

/**
 * @brief Fills a range with uniformly distributed random ExMy (exponent-mantissa) scale values.
 *
 * Accepts human-readable float bounds and maps them to the raw byte range of the
 * target scale type by re-centering the IEEE 754 exponent into the type's own
 * bias space. Sampling is then uniform over raw bytes, which is uniform over
 * representable values of the type.
 *
 * @tparam ScaleType An ExMy scale type (e.g. e8m0_t, e4m3_t, e5m3_t).
 *
 * @note Both bounds snap down to the nearest representable power-of-two in ScaleType space.
 *       If min_scale_ is not an exact power-of-two, the effective lower bound is lower than
 *       requested; if max_scale_ is not an exact power-of-two, the effective upper bound is
 *       also lower than requested.
 *
 * Fields: min_scale_ (lower float bound), max_scale_ (upper float bound, no value
 * generated exceeds it), seed_ (RNG seed; nullopt for random device, default 11939).
 * Precondition: min_scale_ <= max_scale_. Violating this is undefined behavior.
 * Usage:
 *   FillUniformScaleDistribution<e8m0_t>{0.125f, 2.0f, 42}(scale_tensor);
 *   FillUniformScaleDistribution<e4m3_t>{0.125f, 2.0f}(buf.begin(), buf.end());
 */
template <typename ScaleType>
struct FillUniformScaleDistribution
{
    float min_scale_{0.125f};
    float max_scale_{2.0f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        using RawType = typename ScaleType::type; // uint8_t for all current ExMy types

        // Bias and mantissa layout for the target type, resolved at compile time.
        constexpr int float_bias = 127; // IEEE 754 single-precision bias
        constexpr int type_bias =
            numeric_traits<ScaleType>::bias; // e.g. 127 (e8m0), 7 (e4m3), 15 (e5m3)
        constexpr int mant_bits =
            numeric_traits<ScaleType>::mant; // mantissa bits: 0 (e8m0), 3 (e4m3/e5m3)

        // Extract the biased IEEE 754 exponent byte from each float bound.
        // get_exponent(f) == (bit_cast<uint32_t>(f) >> 23) & 0xFF - the raw 8-bit exponent field.
        // Non-power-of-two values snap down: get_exponent(0.1f) == get_exponent(0.0625f) == 123.
        const int ieee_min = static_cast<int>(numeric_utils<float>::get_exponent(min_scale_));
        const int ieee_max = static_cast<int>(numeric_utils<float>::get_exponent(max_scale_));

        // Absolute limits of the raw byte space for this type.
        // raw=0 is reserved: denorm-zero for e4m3/e5m3 (decodes to 0.0), and subnormal
        // territory for e8m0 (2^-127) - excluded to keep all generated values usable as scales.
        // binary_max is the last finite raw value (binary_nan - 1 for all ExMy types).
        constexpr int raw_min = 1;
        constexpr int raw_max = static_cast<int>(numeric<ScaleType>::binary_max);

        // Re-center the IEEE 754 exponent offset into the target type's bias space and pack into
        // raw bytes. (ieee_exp - float_bias) gives the true power: e.g. 123-127 = -4 for 0.0625.
        // Adding type_bias maps into the target encoding: e.g. -4+7 = 3 for e4m3.
        // Left-shift by mant_bits places the exponent field: e.g. 3<<3 = 24 for e4m3.
        // max_r uses mant=0 (not | mant_mask) so it decodes to exactly max_scale - the
        // power-of-two itself. This ensures no generated value exceeds max_scale_ in float space.
        // std::max/min clamp to the valid byte range, preventing out-of-range or NaN raw values.
        const int min_r =
            std::max(((ieee_min - float_bias) + type_bias) * (1 << mant_bits), raw_min);
        const int max_r =
            std::min(((ieee_max - float_bias) + type_bias) * (1 << mant_bits), raw_max);

        // Precondition: clamping must not invert the range. This can happen when both bounds
        // exceed the type's representable range in the same direction (both too large or too
        // small). If triggered, use bounds within the type's representable range.
        assert(min_r <= max_r);

        // Sample raw bytes uniformly in [min_r, max_r], then construct ScaleType directly
        // from the raw byte - bypassing the float ctor which would discard mantissa bits.
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_int_distribution<int> dist(min_r, max_r);
        std::generate(first, last, [&]() { return ScaleType(static_cast<RawType>(dist(gen))); });
    }

    // Range overload: accepts any container or HostTensor with begin()/end().
    template <typename ForwardRange>
    void operator()(ForwardRange&& range) const
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

//----------------------------------------------------------------------------------------------
/// @brief      Transforms given input to fit 2:4 structured sparsity pattern so
///             every subgroup of 4 elements contain at most 2 non-zero elements
template <typename T>
struct AdjustToStructuredSparsity
{
    size_t start{0};
    // masks represent all valid 2:4 structured sparsity permutations
    // clang-format off
    static constexpr int32_t masks[] = {0, 0, 1, 1,
                                        0, 1, 0, 1,
                                        0, 1, 1, 0,
                                        1, 0, 0, 1,
                                        1, 0, 1, 0,
                                        1, 1, 0, 0,
                                        0, 0, 0, 1,
                                        0, 0, 1, 0,
                                        0, 1, 0, 0,
                                        1, 0, 0, 0};
    // clang-format on

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::transform(first, last, first, [=, *this, index = start](T val) mutable {
            auto tmp = val * masks[index % (sizeof(masks) / sizeof(int32_t))];
            index += 1;

            return type_convert<T>(tmp);
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const AdjustToStructuredSparsity&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T, bool UseCos = true, bool UseAbs = false>
struct FillTrigValue
{
    template <typename T_, bool UseCos_ = true, bool UseAbs_ = false>
    struct LinearTrigGen
    {
        int i{0};
        auto operator()()
        {
            float v = 0;
            if constexpr(UseCos_)
            {
                v = cos(i);
            }
            else
            {
                v = sin(i);
            }
            if constexpr(UseAbs_)
                v = abs(v);
            i++;
            return ck_tile::type_convert<T_>(v);
        }
    };
    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        LinearTrigGen<T, UseCos, UseAbs> gen;
        std::generate(first, last, gen);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillTrigValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

} // namespace ck_tile
