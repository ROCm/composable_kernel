// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/multi_index.hpp"

namespace ck {

enum struct IndexTransformEnum
{
    Undefined,
    PassThrough,
    Pad,
    Embed,
    Merge,
    UnMerge,
    Replicate,
    Xor,
    Offset,
};

template <index_t NDimLow, index_t NDimUp>
struct BaseTransform
{
    __host__ __device__ static constexpr auto GetTypeEnum()
    {
        return IndexTransformEnum::Undefined;
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    // return safe value for vector length/stride, based on compile-time known only
    // variables
    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths&, const LowVectorStrides&)
    {
        if constexpr(NDimUp > 0)
        {
            Array<index_t, NDimUp> up_vector_lengths{-1};
            Array<index_t, NDimUp> up_vector_strides{-1};

            return make_tuple(up_vector_lengths, up_vector_strides);
        }
        else
        {
            return make_tuple(Array<index_t, 0>{}, Array<index_t, 0>{});
        }
    }
};

template <typename LowLength>
struct PassThrough : public BaseTransform<1, 1>
{
    static constexpr auto type_enum = IndexTransformEnum::PassThrough;

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr PassThrough() = default;

    __host__ __device__ constexpr PassThrough(const LowLength& low_length)
        : up_lengths_{make_tuple(low_length)}
    {
    }

    __host__ __device__ static constexpr auto GetTypeEnum()
    {
        return IndexTransformEnum::PassThrough;
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("PassThrough{");

        //
        printf("up_lengths_:");
        print(up_lengths_);

        //
        printf("}");
    }
};

template <typename LowLength,
          typename LeftPadLength,
          typename RightPadLength,
          bool SkipIsValidCheck = false>
struct Pad : public BaseTransform<1, 1>
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr Pad() : up_lengths_{}, left_pad_length_{}, right_pad_length_{} {}

    __host__ __device__ constexpr Pad(const LowLength& low_length,
                                      const LeftPadLength& left_pad_length,
                                      const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length + right_pad_length)},
          left_pad_length_{left_pad_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck ||
               ((idx_up[Number<0>{}] >= left_pad_length_) &&
                (idx_up[Number<0>{}] < up_lengths_[Number<0>{}] - right_pad_length_));
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Pad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("left_pad_length_: ");
        print(left_pad_length_);
        printf(", ");

        //
        printf("right_pad_length_: ");
        print(right_pad_length_);

        printf("}");
    }
};

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
struct LeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;

    __host__ __device__ constexpr LeftPad() = default;

    __host__ __device__ constexpr LeftPad(const LowLength& low_length,
                                          const LeftPadLength& left_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length)}, left_pad_length_{left_pad_length}
    {
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] >= left_pad_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        // TODO: we allow pass through this vector length. If one need per-pixel check,
        //       should change the guaranteed vector length while creating the tensor view.
        //       It's up to runtime to check the padding length should be multiple of vector length
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("LeftPad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("left_pad_length_: ");
        print(left_pad_length_);

        printf("}");
    }
};

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
struct RightPad : public BaseTransform<1, 1>
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LowLength low_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr RightPad() = default;

    __host__ __device__ constexpr RightPad(const LowLength& low_length,
                                           const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + right_pad_length)},
          low_length_{low_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] < low_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LowLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        // TODO: we allow pass through this vector length. If one need per-pixel check,
        //       should change the guaranteed vector length while creating the tensor view.
        //       It's up to runtime to check the padding length should be multiple of vector length
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("RightPad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("right_pad_length_: ");
        print(right_pad_length_);

        printf("}");
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
// UpLengths and Coefficients can be either of the followings:
//   1) Tuple of index_t, which is known at run-time, or
//   2) Tuple of Number, which is known at compile-time, or
//   3) Tuple of mixture of index_t and Number, which is known partially at run-time and partially
//   at compile-time
template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct Embed : public BaseTransform<1, UpLengths::Size()>
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    __host__ __device__ constexpr Embed() = default;

    __host__ __device__ constexpr Embed(const UpLengths& up_lengths,
                                        const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    __host__ __device__ static constexpr auto GetTypeEnum() { return IndexTransformEnum::Embed; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Embed{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("coefficients_: ");
        print(coefficients_);

        printf("}");
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_divisor
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return MagicDivision::CalculateMagicNumbers(LowLengths{}[i]);
    }
};

// Implementation of "Merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For Merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct Merge_v2_magic_division : public BaseTransform<LowLengths::Size(), 1>
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    using LowLengthsMagicDivisor = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_divisor<LowLengths>{},
        Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsMagicDivisor low_lengths_magic_divisor_;
    UpLengths up_lengths_;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ constexpr Merge_v2_magic_division() = default;

    __host__ __device__ constexpr Merge_v2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_magic_divisor_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicNumbers(low_lengths[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, I1))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr auto GetTypeEnum() { return IndexTransformEnum::Merge; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[I0];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 = MagicDivision::DoMagicDivision(tmp,
                                                          this->low_lengths_magic_divisor_[i][I0],
                                                          this->low_lengths_magic_divisor_[i][I1]);
            idx_low(i)   = tmp - tmp2 * this->low_lengths_[i];
            tmp          = tmp2;
        });

        idx_low(Number<0>{}) = tmp;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up_new[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 = MagicDivision::DoMagicDivision(tmp,
                                                          this->low_lengths_magic_divisor_[i][I0],
                                                          this->low_lengths_magic_divisor_[i][I1]);

            index_t idx_low_old = idx_low[i];

            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;

            idx_diff_low(i) = idx_low[i] - idx_low_old;
        });

        idx_diff_low(Number<0>{}) = tmp - idx_low(Number<0>{});

        idx_low(Number<0>{}) = tmp;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisor>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        Array<index_t, 1> up_vector_lengths{-1};
        Array<index_t, 1> up_vector_strides{-1};

        up_vector_lengths(0) = low_vector_lengths[Number<NDimLow - 1>{}];
        up_vector_strides(0) = low_vector_strides[Number<NDimLow - 1>{}];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("Merge_v2_magic_division{");

        //
        printf("low_lengths_ ");
        print(low_lengths_);
        printf(", ");

        //
        printf("up_lengths_ ");
        print(up_lengths_);

        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses division and mod. It is supposed to
// be used for low_lengths that are known at compile time and are power of 2, otherwise performance
// will be very bad
template <typename LowLengths>
struct Merge_v3_division_mod : public BaseTransform<LowLengths::Size(), 1>
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v3_division_mod() = default;

    __host__ __device__ constexpr Merge_v3_division_mod(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        // division and mod
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp %= this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0   = Number<0>{};
        constexpr auto INm1 = Number<NDimLow - 1>{};

        index_t tmp = idx_up_new[I0];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            const index_t tmp2 = idx_low[i];
            idx_low(i)         = tmp / this->low_lengths_scan_[i];
            idx_diff_low(i)    = idx_low[i] - tmp2;
            tmp %= this->low_lengths_scan_[i];
        });

        const index_t tmp2 = idx_low[INm1];
        idx_low(INm1)      = tmp;
        idx_diff_low(INm1) = idx_low[INm1] - tmp2;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        Array<index_t, 1> up_vector_lengths{-1};
        Array<index_t, 1> up_vector_strides{-1};

        up_vector_lengths(0) = low_vector_lengths[Number<NDimLow - 1>{}];
        up_vector_strides(0) = low_vector_strides[Number<NDimLow - 1>{}];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("Merge_v3_direct_division_mod{");

        //
        printf("low_lengths_ ");
        print(low_lengths_);
        printf(", ");

        //
        printf("low_lengths_scan_ ");
        print(low_lengths_scan_);
        printf(", ");

        //
        printf("up_lengths_ ");
        print(up_lengths_);

        printf("}");
    }
};

template <typename UpLengths, bool Use24BitIntegerCalculation>
struct UnMerge : public BaseTransform<1, UpLengths::Size()>
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    using UpLengthsScan =
        decltype(container_reverse_exclusive_scan(UpLengths{}, math::multiplies{}, Number<1>{}));

    UpLengths up_lengths_;
    UpLengthsScan up_lengths_scan_;

    __host__ __device__ constexpr UnMerge() = default;

    __host__ __device__ constexpr UnMerge(const UpLengths& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              container_reverse_exclusive_scan(up_lengths, math::multiplies{}, Number<1>{})}
    {
    }

    __host__ __device__ static constexpr auto GetTypeEnum() { return IndexTransformEnum::UnMerge; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        if constexpr(!Use24BitIntegerCalculation)
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}(
                [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
        }
        else
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}([&](auto i) {
                idx_low(Number<0>{}) =
                    (0x00ffffff & idx_low[Number<0>{}]) +
                    (0x00ffffff & idx_up[i]) * (0x00ffffff & up_lengths_scan_[i]);
            });
        }
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&) const
    {
        CalculateLowerIndex(idx_diff_low, idx_diff_up);

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<UpLengthsScan>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ static constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides)
    {
        Array<index_t, NDimUp> up_vector_lengths{-1};
        Array<index_t, NDimUp> up_vector_strides{-1};

        constexpr auto up_length_last = UpLengths{}[Number<NDimUp - 1>{}];

        if constexpr(is_known_at_compile_time<decltype(up_length_last)>::value)
        {
            if(low_vector_lengths[0] != -1)
            {
                up_vector_lengths(NDimUp - 1) = math::gcd(low_vector_lengths[0], up_length_last);
            }
        }

        up_vector_strides(NDimUp - 1) = low_vector_strides[0];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("UnMerge{");

        //
        printf("up_lengths_");
        print(up_lengths_);
        printf(", ");

        //
        printf("up_lengths_scan_");
        print(up_lengths_scan_);

        printf("}");
    }
};

template <typename LowerIndex>
struct Freeze : public BaseTransform<1, 0>
{
    LowerIndex low_idx_;

    __host__ __device__ constexpr Freeze() = default;

    __host__ __device__ constexpr Freeze(const LowerIndex& low_idx) : low_idx_{low_idx} {}

    __host__ __device__ static constexpr auto GetUpperLengths() { return Tuple<>{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& /* idx_up */) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 0,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = low_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& /* idx_diff_up */,
                                                     LowIdx& /* idx_low */,
                                                     const UpIdx& /* idx_up_new */)
    {
        idx_diff_low(Number<0>{}) = 0;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowerIndex>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Freeze{");

        //
        printf("low_idx_: ");
        print(low_idx_);

        printf("}");
    }
};

// Insert a dangling upper dimension without lower dimension
template <typename UpperLength>
struct Insert : public BaseTransform<0, 1>
{
    using UpLengths = decltype(make_tuple(UpperLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr Insert() = default;

    __host__ __device__ constexpr Insert(const UpperLength& up_length)
        : up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 0; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::Size() == 0 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void
    UpdateLowerIndex(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 0 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 0 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpperLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Insert{");

        //
        print(up_lengths_);

        printf("}");
    }
};

// Replicate the original tensor and create a higher dimensional tensor
template <typename UpLengths>
struct Replicate : public BaseTransform<0, UpLengths::Size()>
{
    static constexpr index_t NDimUp = UpLengths::Size();

    __host__ __device__ constexpr Replicate() = default;

    __host__ __device__ constexpr Replicate(const UpLengths& up_lengths) : up_lengths_{up_lengths}
    {
    }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::Size() == 0 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void
    UpdateLowerIndex(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 0 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 0 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Replicate{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);

        printf("}");
    }

    //
    UpLengths up_lengths_;
};

template <typename LowLength, typename SliceBegin, typename SliceEnd>
struct Slice : public BaseTransform<1, 1>
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(SliceEnd{} - SliceBegin{}));

    UpLengths up_lengths_;
    SliceBegin slice_begin_;
    SliceEnd slice_end_;

    __host__ __device__ constexpr Slice() = default;

    __host__ __device__ constexpr Slice(const LowLength&,
                                        const SliceBegin& slice_begin,
                                        const SliceEnd& slice_end)
        : up_lengths_{make_tuple(slice_end - slice_begin)},
          slice_begin_{slice_begin},
          slice_end_{slice_end}
    {
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] + slice_begin_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx&) const
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<SliceBegin>::value &&
               is_known_at_compile_time<SliceEnd>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Slice{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("slice_begin_: ");
        print(slice_begin_);
        printf(", ");

        //
        printf("slice_end_: ");
        print(slice_end_);

        printf("}");
    } // namespace ck
};    // namespace ck

/*
 * \brief lower_idx = upper_idx % modulus.
 * TODO: Need an improved implementation since the modulo operation is expensive.
 */
template <typename Modulus, typename UpLength>
struct Modulo : public BaseTransform<1, 1>
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;
    using UpLengths  = decltype(make_tuple(UpLength{}));

    Modulus modulus_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Modulo() = default;

    __host__ __device__ constexpr Modulo(const Modulus& modulus, const UpLength& up_length)
        : modulus_{modulus}, up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] % modulus_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& up_idx) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        const auto idx_low_old = idx_low;
        idx_low(I0)            = (up_idx(I0) + idx_diff_up(I0)) % modulus_;
        idx_diff_low(I0)       = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Modulus{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);

        printf("}");
    }
};

// 2D XOR
template <typename LowLengths, typename RightShift>
struct Xor : public BaseTransform<2, 2>
{
    static constexpr auto type_enum = IndexTransformEnum::Xor;

    using LowerIndex = MultiIndex<2>;
    using UpperIndex = MultiIndex<2>;

    using UpLengths = LowLengths;

    UpLengths up_lengths_;
    RightShift right_shift_;

    __host__ __device__ constexpr Xor() : up_lengths_{}, right_shift_{} {}

    __host__ __device__ constexpr Xor(const LowLengths& low_lengths, const RightShift& right_shift)
        : up_lengths_{low_lengths}, right_shift_{right_shift}
    {
    }

    __host__ __device__ static constexpr auto GetTypeEnum() { return IndexTransformEnum::Xor; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 2 && UpIdx::Size() == 2,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];

        const auto idx_low_1_tmp =
            (idx_up[Number<1>{}] - idx_up[Number<0>{}] * right_shift_) % up_lengths_[Number<1>{}];

        const auto idx_low_1 =
            (idx_low_1_tmp >= 0) ? idx_low_1_tmp : up_lengths_[Number<1>{}] + idx_low_1_tmp;

        idx_low(Number<1>{}) = idx_low_1;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up) const
    {
        static_assert(LowIdxDiff::Size() == 2 && UpIdxDiff::Size() == 2 && LowIdx::Size() == 2 &&
                          UpIdx::Size() == 2,
                      "wrong! inconsistent # of dimension");

        const auto idx_low_old = idx_low;

        CalculateLowerIndex(idx_low, idx_up);

        idx_diff_low = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<RightShift>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    __host__ __device__ constexpr auto
    CalculateUpperDimensionSafeVectorLengthStrides(const LowVectorLengths& low_vector_lengths,
                                                   const LowVectorStrides& low_vector_strides) const
    {
        Array<index_t, 2> up_vector_lengths = low_vector_lengths;
        Array<index_t, 2> up_vector_strides = low_vector_strides;

        if constexpr(is_known_at_compile_time<RightShift>::value)
        {
            if(low_vector_lengths[1] != -1)
            {
                up_vector_lengths(1) = math::gcd(low_vector_lengths[1], math::abs(right_shift_));
            }
        }

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    __host__ __device__ void Print() const
    {
        printf("Xor{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("right_shift_: ");
        print(right_shift_);

        printf("}");
    }
};

template <typename LowLength, typename OffsetLength>
struct Offset : public BaseTransform<1, 1>
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;
    OffsetLength offset_length_;

    __host__ __device__ constexpr Offset() = default;

    __host__ __device__ constexpr Offset(const LowLength& low_length,
                                         const OffsetLength& offset_length)
        : up_lengths_{make_tuple(low_length)}, offset_length_{offset_length}
    {
    }

    __host__ __device__ static constexpr auto GetTypeEnum() { return IndexTransformEnum::Offset; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] + offset_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx&) const
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<OffsetLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Offset{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("offset_length_: ");
        print(offset_length_);

        printf("}");
    }
};

} // namespace ck
