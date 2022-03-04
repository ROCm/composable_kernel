#ifndef TENSOR_SPACE_FILLING_CURVE_HPP
#define TENSOR_SPACE_FILLING_CURVE_HPP

#include "math.hpp"
#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "tensor_adaptor.hpp"
#include "statically_indexed_array_multi_index.hpp"
#include "tuple_helper.hpp"

namespace ck {

template <typename TensorLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess> // # of scalars per access in each dimension
struct SpaceFillingCurve
{
    static constexpr index_t nDim = TensorLengths::Size();

    using Index = MultiIndex<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, math::multiplies{}, Number<1>{});

    static constexpr auto access_lengths   = TensorLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    static constexpr auto to_index_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(ordered_access_lengths)),
        make_tuple(typename arithmetic_sequence_gen<0, nDim, 1>::type{}),
        make_tuple(Sequence<0>{}));

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr index_t GetNumOfAccess()
    {
        return reduce_on_sequence(TensorLengths{}, math::multiplies{}, Number<1>{}) /
               ScalarPerVector;
    }

    template <index_t AccessIdx1dBegin, index_t AccessIdx1dEnd>
    static __device__ __host__ constexpr auto GetStepBetween(Number<AccessIdx1dBegin>,
                                                             Number<AccessIdx1dEnd>)
    {
        static_assert(AccessIdx1dBegin >= 0, "1D index should be non-negative");
        static_assert(AccessIdx1dBegin < GetNumOfAccess(), "1D index should be larger than 0");
        static_assert(AccessIdx1dEnd >= 0, "1D index should be non-negative");
        static_assert(AccessIdx1dEnd < GetNumOfAccess(), "1D index should be larger than 0");

        constexpr auto idx_begin = GetIndex(Number<AccessIdx1dBegin>{});
        constexpr auto idx_end   = GetIndex(Number<AccessIdx1dEnd>{});
        return idx_end - idx_begin;
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr auto GetForwardStep(Number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d < GetNumOfAccess(), "1D index should be larger than 0");
        return GetStepBetween(Number<AccessIdx1d>{}, Number<AccessIdx1d + 1>{});
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr auto GetBackwardStep(Number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d > 0, "1D index should be larger than 0");

        return GetStepBetween(Number<AccessIdx1d>{}, Number<AccessIdx1d - 1>{});
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr Index GetIndex(Number<AccessIdx1d>)
    {
#if 0
        /*
         * \todo: TensorAdaptor::CalculateBottomIndex does NOT return constexpr as expected.
         */
        constexpr auto ordered_access_idx = to_index_adaptor.CalculateBottomIndex(make_multi_index(Number<AccessIdx1d>{}));
#else

        constexpr auto access_strides = container_reverse_exclusive_scan(
            ordered_access_lengths, math::multiplies{}, Number<1>{});

        constexpr auto idx_1d = Number<AccessIdx1d>{};
        // Given tensor strides \p access_lengths, and 1D index of space-filling-curve, compute the
        // idim-th element of multidimensional index.
        // All constexpr variables have to be captured by VALUE.
        constexpr auto compute_index = [ idx_1d, access_strides ](auto idim) constexpr
        {
            constexpr auto compute_index_impl = [ idx_1d, access_strides ](auto jdim) constexpr
            {
                auto res = idx_1d.value;
                auto id  = 0;

                static_for<0, jdim.value + 1, 1>{}([&](auto kdim) {
                    id = res / access_strides[kdim].value;
                    res -= id * access_strides[kdim].value;
                });

                return id;
            };

            constexpr auto id = compute_index_impl(idim);
            return Number<id>{};
        };

        constexpr auto ordered_access_idx = generate_tuple(compute_index, Number<nDim>{});
#endif
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto idim) {
                index_t tmp = ordered_access_idx[I0];

                static_for<1, idim, 1>{}(
                    [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });

                forward_sweep_(idim) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate multi-dim tensor index
        auto idx_md = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto idim) {
                ordered_idx(idim) = forward_sweep[idim] ? ordered_access_idx[idim]
                                                        : ordered_access_lengths[idim] - 1 -
                                                              ordered_access_idx[idim];
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   ScalarsPerAccess{};
        }();
        return idx_md;
    }
};

} // namespace ck
#endif
