#ifndef CK_MULTI_INDEX_TRANSFORM_HPP
#define CK_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs... xs)
{
    return MultiIndex<sizeof...(Xs)>(xs...);
}

template <index_t Length>
struct PassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<Length>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Sequence<Length>{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ static constexpr bool
    IsUpperIndexInPaddingArea(const UpperIndex& /* idx_up */)
    {
        return false;
    }
};

// LowLengths: Sequence<...>
template <typename LowLengths, typename LeftPads, typename RightPads>
struct Pad
{
    static constexpr index_t nDim = LowLengths::Size();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return GetLowerLengths() + LeftPads{} + RightPads{};
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up - LeftPads{};
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ constexpr bool IsUpperIndexInPaddingArea(const UpperIndex& idx_up) const
    {
        bool flag = false;

        static_for<0, nDim, 1>{}([&](auto idim) {
            // only check if there is left-padding
            static_if<(LeftPads::At(idim) != 0)>{}(
                [&](auto) { flag = flag || idx_up[idim] < LeftPads::At(idim); });

            // only check if there is right-padding
            static_if<(RightPads::At(idim) != 0)>{}([&](auto) {
                flag = flag || idx_up[idim] >= LeftPads::At(idim) + LowLengths::At(idim);
            });
        });

        return flag;
    }
};

// LowLengths: Sequence<...>
template <typename LowLengths>
struct Merge
{
    static constexpr index_t nDimLow = LowLengths::Size();
    static constexpr index_t nDimUp  = 1;

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return LowLengths{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<accumulate_on_sequence(
            GetLowerLengths(), math::multiplies<index_t>{}, Number<1>{})>{};
    }

    // emulate constexpr lambda
    template <typename PseudoLowStrides>
    struct lambda_CalculateLowerIndex
    {
        index_t& itmp;
        LowerIndex& idx_low;

        __host__ __device__ explicit constexpr lambda_CalculateLowerIndex(index_t& itmp_,
                                                                          LowerIndex& idx_low_)
            : itmp(itmp_), idx_low(idx_low_)
        {
        }

        template <typename IDim>
        __host__ __device__ constexpr void operator()(IDim idim) const
        {
            constexpr index_t stride = PseudoLowStrides::At(idim);
            idx_low(idim)            = itmp / stride;
            itmp -= idx_low[idim] * stride;
        }
    };

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        index_t itmp = idx_up[0];

        constexpr auto pseudo_low_strides =
            reverse_inclusive_scan_sequence(
                GetLowerLengths().PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

// calculate index in each of the dimensions in the order of their dimension
#if 1 // would compile to same ISA?
        static_for<0, nDimLow - 1, 1>{}(
            lambda_CalculateLowerIndex<decltype(pseudo_low_strides)>(itmp, idx_low));

        idx_low(nDimLow - 1) = itmp / pseudo_low_strides[nDimLow - 1];
#else
        static_for<0, nDimLow, 1>{}(
            lambda_CalculateLowerIndex<decltype(pseudo_low_strides)>(itmp, idx_low));
#endif

        return idx_low;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        // not implemeneted

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ static constexpr bool
    IsUpperIndexInPaddingArea(const UpperIndex& /* idx_up */)
    {
        return false;
    }
};

// UpLengths: Sequence<...>
template <typename UpLengths>
struct Unmerge
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths()
    {
        constexpr index_t low_length =
            accumulate_on_sequence(UpLengths{}, math::multiplies<index_t>{}, Number<1>{});

        return Sequence<low_length>{};
    }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low{0};

        constexpr auto pseudo_up_strides =
            typename sequence_reverse_inclusive_scan<UpLengths, math::multiplies<index_t>, 1>::
                type{};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low(0) += idx_up[idim] * pseudo_up_strides[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return CalculateLowerIndex(idx_up_diff);
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

// UpLengths: Sequence<...>
// Coefficients: Sequence<...>
// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1] + coefficients[nDimUp]
template <index_t LowLength, typename UpLengths, typename Coefficients>
struct Embed
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ explicit constexpr Embed()
    {
        static_assert(UpLengths::GetSize() == nDimUp && Coefficients::GetSize() == nDimUp + 1,
                      "wrong! # of dimensions not consistent");

        constexpr index_t low_id_max =
            Coefficients::Back() + accumulate_on_sequence(UpLengths{} * Coefficients::PopBack(),
                                                          math::plus<index_t>{},
                                                          Number<0>{});

        static_assert(low_id_max < LowLength, "wrong! lower-id will go out of range");
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetLowerLengths() { return Sequence<LowLength>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low(Coefficients{}[nDimUp]);

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low[0] += idx_up[idim] * Coefficients{}[idim]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        LowerIndex idx_low_diff{0};

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low_diff[0] += idx_up_diff[idim] * Coefficients{}[idim]; });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }
};

} // namespace ck
#endif
