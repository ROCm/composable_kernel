#include "math.hpp"
#include "sequence.hpp"
#include "tensor_adaptor.hpp"
#include "statically_indexed_array_multi_index.hpp"
#include "tuple_helper.hpp"
#include "enable_if.hpp"

namespace ck {

enum TraversalPattern
{
    UShape = 0,
    ZigZag
};

template <typename TensorDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess, // # of scalars per access in each dimension
          enable_if_t<TensorDesc::IsKnownAtCompileTime(), bool>                             = false,
          enable_if_t<TensorDesc::GetNumOfVisibleDimension() == SliceLengths::Size(), bool> = false>
struct SpaceFillingCurve
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index     = MultiIndex<nDim>;
    using Coord     = decltype(make_tensor_coordinate(TensorDesc{}, Index{}));
    using CoordStep = decltype(make_tensor_coordinate_step(TensorDesc{}, Index{}));

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
        // return reduce_on_sequence(
        //     SliceLengths{} / ScalarsPerAccess{}, math::multiplies{}, Number<1>{});
    }

#if 0
    template <int ii>
    static __device__ __host__ Index GetIndex(const Number<ii> Id)
    {
        const auto ordered_access_idx = to_index_adaptor.CalculateBottomIndex(make_multi_index(Id));

        auto forward_sweep = [&]() {
            // auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_access_idx[I0];

                static_for<1, i, 1>{}(
                    [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate data index
        auto data_idx = [&]() {
            // constexpr auto data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i]
                                     ? ordered_access_idx[i]
                                     : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   ScalarsPerAccess{};
        }();
        return data_idx;
    }

#else
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
#endif
};

#if 0
template <typename DataType,
          typename TensorDesc,
          typename TensorBuffer,
          typename SliceLengths,
          typename DimAccessOrder,
          // index_t ScalarPerVector,
          typename ScalarsPerAccess,
          TraversalPattern Pattern                                   = TraversalPattern::UShape,
          std::enable_if_t<TensorDesc::IsKnownAtCompileTime(), bool> = false,
          std::enable_if_t<TensorDesc::GetNumOfVisibleDimension() == SliceLengths::Size(), bool> =
              false>
struct TensorIterator
{
    private:
    public:
    static constexpr index_t nDim = SliceLengths::Size();

    using Index     = MultiIndex<nDim>;
    using Coord     = decltype(make_tensor_coordinate(TensorDesc{}, Index{}));
    using CoordStep = decltype(make_tensor_coordinate_step(TensorDesc{}, Index{}));
    // static constexpr index_t ScalarPerVector =
    //     reduce_on_sequence(ScalarsPerAccess{}, math::multiplies{}, Number<1>{});

    static constexpr index_t ScalarPerVector = reduce_on_sequence(
        transform_sequences([](auto step) { return step == 0 ? 1 : step; }, ScalarsPerAccess{}),
        math::multiplies{},
        Number<1>{});

    using Vector       = vector_type_maker_t<DataType, ScalarPerVector>;
    using src_vector_t = typename Vector::type;

    __host__ __device__ constexpr TensorIterator(TensorBuffer& tensor_buf)
        : tensor_buf_(tensor_buf), coord_()
    {

        static_assert(TensorDesc::IsKnownAtCompileTime(),
                      "wrong! TensorDesc needs to be compile-time const");
        // assert modulos are zero in all dimensions
        // static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % ScalarPerVector == 0,
        // "wrong!");
    }

    __host__ __device__ src_vector_t Get(index_t i) const
    {
        const bool is_src_valid =
            coordinate_has_valid_offset_assuming_visible_index_is_valid(desc_, coord_);

        // return tensor_buf_.template Get<src_vector_t>(coord_.GetOffset(), is_src_valid);
        // return Get<src_vector_t>(tensor_buf_, coord_.GetOffset(), is_src_valid);
        return tensor_buf_.template Get<src_vector_t>(desc_.CalculateOffset(coord_.GetIndex()),
                                                      is_src_valid);
    }

    __host__ __device__ src_vector_t Set(const src_vector_t& x)
    {
        const bool is_src_valid =
            coordinate_has_valid_offset_assuming_visible_index_is_valid(desc_, coord_);

        return tensor_buf_.template Set<src_vector_t>(coord_.GetOffset(), is_src_valid, x);
        // Set<src_vector_t>(tensor_buf_, coord_.GetOffset(), is_src_valid, x);
    }

    __device__ __host__ TensorIterator* operator++()
    {
        // {
        //     const Index idx = coord_.GetIndex();
        //     printf("init coord_.GetIndex() = %d, %d, %d\n",
        //            idx[Number<0>{}],
        //            idx[Number<1>{}],
        //            idx[Number<2>{}]);
        // }

        constexpr auto desc = TensorDesc{};
        auto cur_index      = coord_.GetIndex();

        auto next_index = to_multi_index(coord_.GetIndex()) + ScalarsPerAccess{};

        constexpr auto slice_lengths = to_multi_index(SliceLengths{});

#if 0
        auto compute_next_index = [&next_index, &slice_lengths, &cur_index ](auto ii) constexpr
        {
            if(next_index[ii] >= slice_lengths[ii])
            {
                next_index(ii) = next_index[ii] - slice_lengths[ii];
                // next_index(ii + Number<1>{}) +=
                //     (next_index[ii] - slice_lengths[ii]) / slice_lengths[ii];
                next_index(ii + Number<1>{}) += 1;
            }
        };
        static_for<0, nDim - 1, 1>{}(compute_next_index);

        const auto step =
            make_tensor_coordinate_step(desc, next_index - to_multi_index(coord_.GetIndex()));
#else

        Index step_index        = to_multi_index(ScalarsPerAccess{});
        auto compute_step_index = [&slice_lengths, &cur_index, &step_index ](auto ii) constexpr
        {
            if(cur_index[ii] == slice_lengths[ii] - 1)
            {
                step_index(ii) = step_index[ii] - slice_lengths[ii] + 1;
                step_index(ii + Number<1>{}) += 1;
            }
            step_index(ii) = (step_index[ii] + cur_index[ii]) % slice_lengths[ii];
            step_index(ii + Number<1>{}) += (step_index[ii] + cur_index[ii]) / slice_lengths[ii];
        };
        static_for<0, nDim - 1, 1>{}(compute_step_index);

        const auto step = make_tensor_coordinate_step(desc, step_index);
#endif
        {
            const Index idx = step.GetIndexDiff();
            // printf("step.GetIndex() = %d, %d, %d\n",
            //        idx[Number<0>{}],
            //        // idx[Number<1>{}],
            //        // idx[Number<2>{}]);
            //        idx[Number<1>{}]);
        }

        move_tensor_coordinate(desc, coord_, step);
        {
            const Index idx = coord_.GetIndex();
            // printf("coord_.GetIndex() = %d, %d, %d\n",
            //        idx[Number<0>{}],
            //        // idx[Number<1>{}],
            //        // idx[Number<2>{}]);
            //        idx[Number<1>{}],
            // idx[Number<2>{}]);
        }

        return this;
    }

    template <typename SliceMoveStepIdx>
    __device__ void MoveSrcSliceWindow(const SliceMoveStepIdx& slice_move_step_idx)
    {
        constexpr auto desc = TensorDesc{};

        const auto src_slice_move_step_iter =
            make_tensor_coordinate_step(desc, to_multi_index(slice_move_step_idx));

        move_tensor_coordinate(desc, coord_, src_slice_move_step_iter);
    }

    private:
    static constexpr auto desc_ = remove_cvref_t<TensorDesc>{}; // use reference for dynamic tensor
    TensorBuffer& tensor_buf_;
    Coord coord_;
};
#endif

} // namespace ck
