#include "math.hpp"
#include "sequence.hpp"
#include "tensor_adaptor.hpp"
#include "statically_indexed_array_multi_index.hpp"
#include "tuple_helper.hpp"

namespace ck {

template <typename SliceLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess> // # of scalars per access in each dimension
struct SpaceFillingCurve
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, math::multiplies{}, Number<1>{});

    static constexpr auto access_lengths   = SliceLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    static constexpr auto to_index_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(ordered_access_lengths)),
        make_tuple(typename arithmetic_sequence_gen<0, nDim, 1>::type{}),
        make_tuple(Sequence<0>{}));

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __device__ __host__ static constexpr index_t GetNumOfAccess()
    {
        return reduce_on_sequence(SliceLengths{}, math::multiplies{}, Number<1>{}) /
               ScalarPerVector;
    }

    template <int ii>
    static __device__ __host__ constexpr Index GetIndex(Number<ii>)
    {
#if 0
        /*
         * \todo: TensorAdaptor::CalculateBottomIndex does NOT return constexpr as expected.
         */
        constexpr auto ordered_access_idx = to_index_adaptor.CalculateBottomIndex(make_multi_index(Number<ii>{}));
#else

        constexpr auto access_strides = container_reverse_exclusive_scan(
            ordered_access_lengths, math::multiplies{}, Number<1>{});

        constexpr auto Id = Number<ii>{};
        // all constexpr has to be captured by VALUE.
        constexpr auto compute_index = [ Id, access_strides ](auto idim) constexpr
        {
            constexpr auto compute_index_impl = [ Id, access_strides ](auto jdim) constexpr
            {
                auto res = Id.value;
                auto id  = 0;

                static_for<0, jdim.value + 1, 1>{}([&](auto jj) {
                    id = res / access_strides[jj].value;
                    res -= id * access_strides[jj].value;
                });

                return id;
            };

            constexpr auto id = compute_index_impl(idim);
            return Number<id>{};
        };

        constexpr auto ordered_access_idx = generate_tuple(compute_index, Number<nDim>{});
#endif
        constexpr auto forward_sweep = [&]() {
            // auto forward_sweep = [&]() {
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

        // calculate data index
        auto data_idx = [&]() {
            // constexpr auto data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto idim) {
                ordered_idx(idim) = forward_sweep[idim] ? ordered_access_idx[idim]
                                                        : ordered_access_lengths[idim] - 1 -
                                                              ordered_access_idx[idim];
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   ScalarsPerAccess{};
        }();
        return data_idx;
    }
};

} // namespace ck
