#ifndef CK_DIMENSION_TRANSFORM_HPP
#define CK_DIMENSION_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = Array<index_t, N>;

// LowLengths: Sequence<...>
template <class LowLengths>
struct PassThrough
{
    static constexpr index_t nDim = LowLengths::GetSize();

    using LowerId = MultiIndex<nDim>;
    using UpperId = LowerId;

    __host__ __device__ static constexpr auto GetLowerNumOfDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetUpperNumOfDimension()
    {
        return GetLowerNumOfDimension();
    }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return GetLowerLengths(); }

    __host__ __device__ static constexpr auto GetLowerId(UpperId id_up) { return id_up; }

    __host__ __device__ static constexpr auto GetLowerIdDiff(UpperId id_up_diff)
    {
        return id_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

// LowLengths: Sequence<...>
template <class LowLengths, class LeftPads, class RightPads>
struct Pad
{
    static constexpr index_t nDim = LowLengths::GetSize();

    using LowerId = MultiIndex<nDim>;
    using UpperId = LowerId;

    __host__ __device__ static constexpr auto GetLowerNumOfDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetUpperNumOfDimension()
    {
        return GetLowerNumOfDimension();
    }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return GetLowerLengths() + LeftPads + RightPads;
    }

    __host__ __device__ static constexpr auto GetLowerId(UpperId id_up) { return id_up - LeftPads; }

    __host__ __device__ static constexpr auto GetLowerIdDiff(UpperId id_up_diff)
    {
        return id_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

// LowLengths: Sequence<...>
template <class LowLengths>
struct Merge
{
    static constexpr index_t nDimLow = LowLengths::GetSize();
    static constexpr index_t nDimUp  = 1;

    using LowerId = MultiIndex<nDimLow>;
    using UpperId = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetUpperNumOfDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetLowerNumOfDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<accumulate_on_sequence(
            GetLowerLengths(), math::multiplies<index_t>{}, Number<1>{})>{};
    }

    __host__ __device__ static constexpr auto GetLowerId(UpperId id_up)
    {
        LowerId id_low;

        // not implemeneted

        return id_low;
    }

    // id_low_diff depends on id_low_old, so id_low need to be up-to-date
    __host__ __device__ static constexpr auto GetLowerIdDiff(UpperId id_up_diff, LowerId id_low_old)
    {
        LowerId id_low_diff;

        // not implemeneted

        return id_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }
};

// UpLengths: Sequence<...>
template <index_t LowLength, class UpLengths>
struct Unmerge
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpLengths::GetSize();

    __host__ __device__ constexpr Unmerge()
    {
        static_assert(LowLength == accumulate_on_sequence(
                                       UpLengths{}, math::multiplies<index_t>{}, Number<1>{}),
                      "wrong! UpLengths need to be ");
    }

    __host__ __device__ static constexpr auto GetUpperNumOfDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetLowerNumOfDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<LowLength>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto GetLowerId(UpperId id_up)
    {
        constexpr auto scans = typename sequence_reverse_inclusive_scan<UpLengths,
                                                                        math::multiplies<index_t>,
                                                                        1>::type{};

        LowerId id_low{0};

        static_for<0, nDim, 1>{}([&](auto idim) { id_low[0] += id_up[idim] * scans[idim]; });

        return id_low;
    }

    __host__ __device__ static constexpr auto GetLowerIdDiff(UpperId id_up_diff)
    {
        return GetLowerId(id_up_diff);
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

// UpLengths: Sequence<...>
// Coefficients: Sequence<...>
// id_low = coefficients[0, ...nDimUp-1] * id_up[0, ...nDimUp-1] + coefficients[nDimUp]
template <index_t LowLength, class UpLengths, class Coefficients>
struct Embed
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpLengths::GetSize();

    static constexpr auto mCoefficients = Coefficients{};

    __host__ __device__ constexpr Embed()
    {
        static_assert(UpLengths::GetSize() == nDimUp && Coefficients::GetSize() == nDimUp + 1,
                      "wrong! # of dimensions not consistent");

        constexpr index_t low_id_max =
            Coefficents.Back() + accumulate_on_sequence(UpLengths{} * Coefficients::PopBack(),
                                                        math::plus<index_t>{},
                                                        Number<0>{});

        static_assert(low_id_max < LowLength, "wrong! lower-id will go out of range");
    }

    __host__ __device__ static constexpr auto GetUpperNumOfDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetLowerNumOfDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<LowLength>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto GetLowerId(UpperId id_up)
    {
        LowerId id_low{mCoefficients[nDimUp]};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { id_low[0] += id_up[idim] * mCoefficients[idim]; });

        return id_low;
    }

    __host__ __device__ static constexpr auto GetLowerIdDiff(UpperId id_up_diff)
    {
        LowerId id_low_diff{0};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { id_low_diff[0] += id_up_diff[idim] * mCoefficients[idim]; });

        return id_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

} // namespace ck
#endif
