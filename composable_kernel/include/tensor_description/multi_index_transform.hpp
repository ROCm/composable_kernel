#ifndef CK_MULTI_INDEX_TRANSFORM_HPP
#define CK_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"
#include "multi_index.hpp"

namespace ck {

template <index_t Length>
struct PassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<1>{}; }

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

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

// By default, will automatically judge if is-valid check for upper-to-lower-index-mapping is
// necessary
// However, the check will be skipped if SkipIsValidCheck is set to true by user
// LowerLengths: Sequence<...>
template <typename LowerLengths,
          typename LeftPads,
          typename RightPads,
          bool SkipIsValidCheck = false>
struct Pad
{
    static constexpr index_t nDim = LowerLengths::Size();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = MultiIndex<nDim>;

    __host__ __device__ constexpr Pad()
    {
        static_assert(LowerLengths::GetSize() == nDim && LeftPads::GetSize() == nDim &&
                          RightPads::GetSize() == nDim,
                      "wrong! # of dimensions not consistent");
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return LowerLengths{} + LeftPads{} + RightPads{};
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

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        // skip valid check if user request it
        if(SkipIsValidCheck)
        {
            return true;
        }

        bool flag = true;

        for(index_t i = 0; i < nDim; ++i)
        {
            flag = flag && LeftPads::At(i) == 0 && RightPads::At(i) == 0;
        }

        return flag;
    }
};

// LowerLengths: Sequence<...>
// SliceBegins: Sequence<...>
// SliceEnds: Sequence<...>
template <typename LowerLengths, typename SliceBegins, typename SliceEnds>
struct Slice
{
    static constexpr index_t nDim = LowerLengths::Size();

    using LowerIndex = MultiIndex<nDim>;
    using UpperIndex = MultiIndex<nDim>;

    __host__ __device__ constexpr Slice()
    {
        static_assert(LowerLengths::GetSize() == nDim && SliceBegins::GetSize() == nDim &&
                          SliceEnds::GetSize() == nDim,
                      "wrong! # of dimensions not consistent");

#if 0 
        // TODO: would not compile, error on constexpr
        static_for<0, nDim, 1>{}([&](auto idim) {
            static_assert(SliceBegins::At(idim) <= SliceEnds::At(idim) &&
                              SliceBegins::At(idim) >= 0 &&
                              SliceEnds::At(idim) <= LowerLengths::At(idim),
                          "wrong! Slice config is wrong");
        });
#endif
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return SliceEnds{} - SliceBegins{};
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up + SliceBegins{};
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

// LowerLengths: Sequence<...>
template <typename LowerLengths>
struct Merge
{
    static constexpr index_t nDimLow = LowerLengths::Size();
    static constexpr index_t nDimUp  = 1;

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Sequence<reduce_on_sequence(
            LowerLengths{}, math::multiplies<index_t>{}, Number<1>{})>{};
    }

    // emulate constexpr lambda
    template <typename PseudoLowStrides>
    struct lambda_CalculateLowerIndex
    {
        index_t& itmp;
        LowerIndex& idx_low;

        __host__ __device__ constexpr lambda_CalculateLowerIndex(index_t& itmp_,
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

        index_t itmp = idx_up[Number<0>{}];

        constexpr auto pseudo_low_strides =
            reverse_inclusive_scan_sequence(
                LowerLengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

        static_for<0, nDimLow - 1, 1>{}(
            lambda_CalculateLowerIndex<decltype(pseudo_low_strides)>(itmp, idx_low));

        idx_low(Number<nDimLow - 1>{}) = itmp / pseudo_low_strides[Number<nDimLow - 1>{}];

        return idx_low;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    // If idx_up_diff is known at compile-time, many calculations can be optimized
    // away by compiler
    // This function assume idx_low_old is not out-of-bound
    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& idx_low_old)
    {
        if(idx_up_diff[Number<0>{}] == 0)
        {
            return make_zero_multi_index<nDimLow>();
        }
        else
        {
            // CalculateLowerIndex(idx_up_diff) has multiple integer divisions.
            //   If idx_up_diff is known at compile-time, the calculation can
            //   be done at compile-time. However, if idx_up_diff is only known
            //   at run-time, then the calculation will also be computed at
            //   run-time, and can be very expensive.
            LowerIndex idx_low_diff_tmp = CalculateLowerIndex(idx_up_diff);

            // find out the last low dimension that changed
            index_t last_changed_low_dim = 0;

            static_for<0, nDimLow, 1>{}([&](auto i) {
                if(idx_low_diff_tmp[i] != 0)
                {
                    last_changed_low_dim = i;
                }
            });

            LowerIndex idx_low_new = idx_low_old + idx_low_diff_tmp;

            if(idx_up_diff[Number<0>{}] > 0)
            {
                // do carry check on each low dimension in reversed order
                // starting from the first digit that changed
                // don't check the highest dimension
                bool carry = false;

                static_for<nDimLow - 1, 0, -1>{}([&](auto i) {
                    if(i <= last_changed_low_dim)
                    {
                        if(carry)
                        {
                            ++idx_low_new(i);
                        }

                        carry = false;

                        if(idx_low_new[i] >= LowerLengths::At(i))
                        {
                            idx_low_new(i) -= LowerLengths::At(i);
                            carry = true;
                        }
                    }
                });

                // highest dimension, no out-of-bound check
                if(carry)
                {
                    ++idx_low_new(Number<0>{});
                }
            }
            else
            {
                // do borrow check on each low dimension in reversed order
                // starting from the first digit that changed
                // don't check the highest dimension
                bool borrow = false;

                static_for<nDimLow - 1, 0, -1>{}([&](auto i) {
                    if(i <= last_changed_low_dim)
                    {
                        if(borrow)
                        {
                            --idx_low_new(i);
                        }

                        borrow = false;

                        if(idx_low_new[i] < 0)
                        {
                            idx_low_new(i) += LowerLengths::At(i);
                            borrow = true;
                        }
                    }
                });

                // highest dimension, no out-of-bound check
                if(borrow)
                {
                    --idx_low_new(Number<0>{});
                }
            }

            return idx_low_new - idx_low_old;
        }
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

// UpperLengths: Sequence<...>
template <typename UpperLengths>
struct UnMerge
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpperLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpperLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low = make_multi_index(0);

        constexpr auto pseudo_up_strides =
            reverse_inclusive_scan_sequence(
                UpperLengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

        static_for<0, nDimUp, 1>{}(
            [&](auto idim) { idx_low(Number<0>{}) += idx_up[idim] * pseudo_up_strides[idim]; });

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

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

// By default, will automatically judge if is-valid check for upper-to-lower-index-mapping is
// necessary
// However, the check will be skipped if SkipIsValidCheck is set to true by user
// UpperLengths: Sequence<...>
// Coefficients: Sequence<...>
// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1] + coefficients[nDimUp]
template <index_t LowerLength,
          typename UpperLengths,
          typename Coefficients,
          bool SkipIsValidCheck = false>
struct Embed
{
    static constexpr index_t nDimLow = 1;
    static constexpr index_t nDimUp  = UpperLengths::Size();

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ constexpr Embed()
    {
        static_assert(UpperLengths::GetSize() == nDimUp && Coefficients::GetSize() == nDimUp + 1,
                      "wrong! # of dimensions not consistent");
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<nDimUp>{}; }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpperLengths{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low = make_multi_index(Coefficients{}[Number<nDimUp>{}]);

        static_for<0, nDimUp, 1>{}(
            [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * Coefficients{}[i]; });

        return idx_low;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        LowerIndex idx_low_diff = make_multi_index(0);

        static_for<0, nDimUp, 1>{}(
            [&](auto i) { idx_low_diff(Number<0>{}) += idx_up_diff[i] * Coefficients{}[i]; });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        // skip valid check if user request it
        if(SkipIsValidCheck)
        {
            return true;
        }

        bool flag = true;

        index_t ncorner = 1;

        for(index_t idim = 0; idim < nDimUp; ++idim)
        {
            ncorner *= 2;
        }

        // loop over each corner of the upper tensor
        for(index_t icorner = 0; icorner < ncorner; ++icorner)
        {
            // generate upper index for each corner
            auto idx_up = make_zero_multi_index<nDimUp>();

            index_t itmp = icorner;

            static_for<nDimUp, 0, -1>{}([&](auto idim) {
                auto idim_m1    = idim - Number<1>{};
                idx_up(idim_m1) = itmp % 2 == 0 ? 0 : UpperLengths::At(idim_m1) - 1;
                itmp /= 2;
            });

            // calculate lower index
            auto idx_low = CalculateLowerIndex(idx_up);

            // judge if lower index is valid
            flag = flag && idx_low[Number<0>{}] >= 0 && idx_low[Number<0>{}] < LowerLength;
        }

        return flag;
    }
};

// LowerLengths: Sequence<...>
// LowerFreezePoint: Sequence<...>
template <typename LowerLengths, typename LowerFreezePoint>
struct Freeze
{
    static constexpr index_t nDimLow = LowerLengths::Size();
    static constexpr index_t nDimUp  = 0;

    using LowerIndex = MultiIndex<nDimLow>;
    using UpperIndex = MultiIndex<nDimUp>;

    __host__ __device__ constexpr Freeze()
    {
        // TODO: sanity check: LowerFreezePoint should be within range of LowerLengths
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<nDimLow>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<0>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Sequence<>{}; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& /*idx_up*/)
    {
        return to_multi_index(LowerFreezePoint{});
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& /* idx_up_diff */,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return make_zero_multi_index<nDimLow>();
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

} // namespace ck
#endif
