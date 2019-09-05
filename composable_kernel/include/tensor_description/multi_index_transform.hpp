#ifndef CK_MULTI_INDEX_TRANSFORM_HPP
#define CK_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = Array<index_t, N>;

// LowLengths: Sequence<...>
template <class LowLengths>
struct PassThrough
{
    static constexpr index_t nDim = LowLengths::GetSize();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = LowerIndex;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        return GetNumOfLowerDimension();
    }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return GetLowerLengths(); }

    __host__ __device__ static constexpr auto GetLowerIndex(UpperIndex idx_up) { return idx_up; }

    __host__ __device__ static constexpr auto GetLowerIndexDiff(UpperIndex idx_up_diff)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsIndexTransformLinear() { return true; }
};

// LowLengths: Sequence<...>
template <class LowLengths, class LeftPads, class RightPads>
struct Pad
{
    static constexpr index_t nDim = LowLengths::GetSize();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = LowerIndex;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        return GetNumOfLowerDimension();
    }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return GetLowerLengths() + LeftPads + RightPads;
    }

    __host__ __device__ static constexpr auto GetLowerIndex(UpperIndex idx_up)
    {
        return idx_up - LeftPads;
    }

    __host__ __device__ static constexpr auto GetLowerIndexDiff(UpperIndex idx_up_diff)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsIndexTransformLinear() { return true; }
};

// LowLengths: Sequence<...>
template <class LowLengths>
struct Merge
{
    static constexpr index_t nDimLow = LowLengths::GetSize();
    static constexpr index_t nDimUp  = 1;

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfUpperDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<accumulate_on_sequence(
            GetLowerLengths(), math::multiplies<index_t>{}, Number<1>{})>{};
    }

    __host__ __device__ static constexpr auto GetLowerIndex(UpperIndex idx_up)
    {
        LowerIndex idx_low;

        // not implemeneted

        return idx_low;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    __host__ __device__ static constexpr auto GetLowerIndexDiff(UpperIndex idx_up_diff,
                                                                LowerIndex idx_low_old)
    {
        LowerIndex idx_low_diff;

        // not implemeneted

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsIndexTransformLinear() { return false; }
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

    __host__ __device__ static constexpr auto GetNumOfUpperDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<LowLength>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto GetLowerIndex(UpperIndex idx_up)
    {
        constexpr auto scans = typename sequence_reverse_inclusive_scan<UpLengths,
                                                                        math::multiplies<index_t>,
                                                                        1>::type{};

        LowerIndex idx_low{0};

        static_for<0, nDim, 1>{}([&](auto idim) { idx_low[0] += idx_up[idim] * scans[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto GetLowerIndexDiff(UpperIndex idx_up_diff)
    {
        return GetLowerIndex(idx_up_diff);
    }

    __host__ __device__ static constexpr bool IsIndexTransformLinear() { return true; }
};

// UpLengths: Sequence<...>
// Coefficients: Sequence<...>
// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1] + coefficients[nDimUp]
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

    __host__ __device__ static constexpr auto GetNumOfUpperDimension(){return Number<nDimUp>{}};

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<LowLength>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto GetLowerIndex(UpperIndex idx_up)
    {
        LowerIndex idx_low{mCoefficients[nDimUp]};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low[0] += idx_up[idim] * mCoefficients[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto GetLowerIndexDiff(UpperIndex idx_up_diff)
    {
        LowerIndex idx_low_diff{0};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low_diff[0] += idx_up_diff[idim] * mCoefficients[idim]; });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsIndexTransformLinear() { return true; }
};

} // namespace ck
#endif
