#pragma once
#include "ck/host/utils.hpp"
#include "ck/host/tuples.hpp"
#include "ck/host/seq.hpp"
namespace ck {
namespace host {

template <typename Container,
          typename Reduce,
          typename ROld,
          index_t I,
          index_t IEnd,
          index_t IStep>
constexpr auto container_reduce_impl(
    const Container& x, Reduce reduce, ROld r_old, Number<I> i, Number<IEnd>, Number<IStep>)
{
    auto r_new = reduce(x[i], r_old);

    if constexpr(i.value < IEnd - IStep)
    {
        return container_reduce_impl(
            x, reduce, r_new, i + Number<IStep>{}, Number<IEnd>{}, Number<IStep>{});
    }
    else
    {
        return r_new;
    }
}

template <typename Container,
          typename Reduce,
          typename Init,
          index_t IBegin = 0,
          index_t IEnd   = Container::Size(),
          index_t IStep  = 1>
constexpr auto container_reduce(const Container& x,
                                Reduce reduce,
                                Init init,
                                Number<IBegin> = Number<0>{},
                                Number<IEnd>   = Number<Container::Size()>{},
                                Number<IStep>  = Number<1>{})
{
    static_assert((IEnd - IBegin) % IStep == 0, "wrong!");

    if constexpr(IEnd > IBegin)
    {
        return container_reduce_impl(
            x, reduce, init, Number<IBegin>{}, Number<IEnd>{}, Number<IStep>{});
    }
    else
    {
        return init;
    }
}

template <index_t NDimHidden, typename VisibleDimensionIds>
struct TensorCoordinate
{
    // TODO make these private
    static constexpr index_t ndim_visible_ = VisibleDimensionIds::Size();

    using HiddenIndex  = MultiIndex<NDimHidden>;
    using VisibleIndex = MultiIndex<ndim_visible_>;

    public:
    constexpr TensorCoordinate() = default;

    constexpr TensorCoordinate(const HiddenIndex& idx_hidden) : idx_hidden_{idx_hidden} {}

    constexpr auto GetIndex() const { return GetVisibleIndex(); }

    constexpr index_t GetOffset() const { return idx_hidden_[Number<0>{}]; }

    // TODO make these private
    constexpr const auto& GetHiddenIndex() const { return idx_hidden_; }

    auto& GetHiddenIndex() { return idx_hidden_; }

    constexpr auto GetVisibleIndex() const
    {
        return get_container_subset(idx_hidden_, VisibleDimensionIds{});
    }

    // TODO make these private
    HiddenIndex idx_hidden_;
};
template <typename Transforms,
          typename LowerDimensionIdss,
          typename UpperDimensionIdss,
          typename VisibleDimensionIds,
          typename ElementSpaceSize>
struct TensorDescriptor
{
    // TODO make these private
    static constexpr index_t GetNumOfTransform() { return Transforms::Size(); }

    static constexpr index_t GetNumOfVisibleDimension() { return VisibleDimensionIds::Size(); }

    static constexpr index_t GetNumOfHiddenDimension()
    {
        constexpr auto all_low_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); }, LowerDimensionIdss{});

        constexpr auto all_up_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); }, UpperDimensionIdss{});

        constexpr auto all_dim_ids = merge_sequences(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = typename sequence_unique_sort<decltype(all_dim_ids),
                                                                      less<index_t>,
                                                                      equal<index_t>>::type;

        return unique_sort_all_dim_ids::Size();
    }

    static constexpr auto InitializeElementSize(const Transforms& transforms)
    {
        const auto lengths = generate_tuple(
            [&](auto idim_visible) {
                constexpr auto tmp = GetTransformAndItsUpperDimension(idim_visible);

                constexpr index_t itran   = tmp[Number<0>{}];
                constexpr index_t idim_up = tmp[Number<1>{}];
                constexpr bool found      = tmp[Number<2>{}];

                static_assert(found == true,
                              "wrong! not found matching transformation and upper-dimension");

                const auto length =
                    transforms[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];

                return length;
            },
            Number<ndim_visible_>{});

        // TODO: make container_reduce support tuple of Number and index_t
        return container_reduce(lengths, multiplies{}, Number<1>{});
    }

    template <index_t IDim>
    static constexpr auto GetTransformAndItsUpperDimension(Number<IDim>)
    {
        constexpr auto idim_visible = Number<IDim>{};

        constexpr index_t idim_hidden = VisibleDimensionIds::At(idim_visible);

        index_t itran_found   = 0;
        index_t idim_up_found = 0;
        bool found            = false;

        static_for<0, ntransform_, 1>{}([&](auto itran) {
            constexpr auto up_dim_ids = UpperDimensionIdss{}[itran];

            static_for<0, up_dim_ids.Size(), 1>{}([&](auto idim_up) {
                if constexpr(up_dim_ids[idim_up] == idim_hidden)
                {
                    itran_found   = itran;
                    idim_up_found = idim_up;
                    found         = true;
                }
            });
        });

        return ck::host::make_tuple(itran_found, idim_up_found, found);
    }

    constexpr static index_t ntransform_   = GetNumOfTransform();
    constexpr static index_t ndim_visible_ = GetNumOfVisibleDimension();
    constexpr static index_t ndim_hidden_  = GetNumOfHiddenDimension();

    using VisibleIndex = MultiIndex<ndim_visible_>;
    using HiddenIndex  = MultiIndex<ndim_hidden_>;
    using Coordinate   = TensorCoordinate<ndim_hidden_, VisibleDimensionIds>;

    // may be index_t or Number<>
    using ElementSize = remove_cv_t<decltype(InitializeElementSize(Transforms{}))>;

    public:
#if 0 // workaround compiler complaint about constexpr
							           constexpr TensorDescriptor() = default;
#else
    constexpr TensorDescriptor() : transforms_{}, element_size_{}, element_space_size_{} {}
#endif

    constexpr TensorDescriptor(const Transforms& transforms, ElementSpaceSize element_space_size)
        : transforms_{transforms},
          element_size_{InitializeElementSize(transforms)},
          element_space_size_{element_space_size}

    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionIdss::Size() == ntransform_ &&
                          UpperDimensionIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    static constexpr index_t GetNumOfDimension() { return GetNumOfVisibleDimension(); }

    template <index_t IDim>
    constexpr auto GetLength(Number<IDim>) const
    {
        static_assert(IDim >= 0 && IDim < ndim_visible_, "wrong! out of range");

        constexpr auto tmp = GetTransformAndItsUpperDimension(Number<IDim>{});

        constexpr index_t itran   = tmp[Number<0>{}];
        constexpr index_t idim_up = tmp[Number<1>{}];
        constexpr bool found      = tmp[Number<2>{}];

        static_assert(found == true,
                      "wrong! not found matching transformation and upper-dimension");

        return transforms_[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];
    }

    constexpr auto GetLengths() const
    {
        // FIXME: use Tuple of reference instead
        return generate_sequence_v2([&](auto I) { return GetLength(I); }, Number<ndim_visible_>{});
    }

    constexpr auto GetElementSize() const { return element_size_; }

    constexpr auto GetElementSpaceSize() const { return element_space_size_; }

    template <typename Idx>
    constexpr index_t CalculateOffset(const Idx& idx) const
    {
        static_assert(Idx::Size() == GetNumOfDimension(), "wrong! inconsistent # of dimension");

        return make_tensor_coordinate(*this, idx).GetOffset();
    }

    // TODO make these private.
    constexpr const auto& GetTransforms() const { return transforms_; }

    static constexpr auto GetLowerDimensionIdss() { return LowerDimensionIdss{}; }

    static constexpr auto GetUpperDimensionIdss() { return UpperDimensionIdss{}; }

    static constexpr auto GetVisibleDimensionIds() { return VisibleDimensionIds{}; }

    static constexpr bool IsKnownAtCompileTime()
    {
        bool is_known = true;

        static_for<0, Transforms::Size(), 1>{}([&](auto i) {
            is_known &= remove_cvref_t<decltype(Transforms{}[i])>::IsKnownAtCompileTime();
        });

        return is_known && is_known_at_compile_time<ElementSize>::value &&
               is_known_at_compile_time<ElementSpaceSize>::value;
    }
    Transforms transforms_;
    ElementSize element_size_;
    ElementSpaceSize element_space_size_;
};
template <typename Lengths, typename Strides, index_t I, typename AccOld>
constexpr auto calculate_element_space_size_impl(const Lengths& lengths,
                                                 const Strides& strides,
                                                 Number<I> i,
                                                 AccOld acc_old)
{
    auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

    if constexpr(i.value < Lengths::Size() - 1)
    {
        return calculate_element_space_size_impl(lengths, strides, i + Number<1>{}, acc_new);
    }
    else
    {
        return acc_new;
    }
}

template <typename... Lengths,
          typename... Strides,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
constexpr auto make_naive_tensor_descriptor(const Tuple<Lengths...>& lengths,
                                            const Tuple<Strides...>& strides)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_embed_transform(lengths, strides));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};
    const auto element_space_size         = calculate_element_space_size_impl(
        lengths,
        strides,
        Number<0>{},
        LongNumber<1>{}); // FIXME: if using index_t, is just Number instead of LongNUmber okay?
    /**auto f = [&](auto fs, auto i, auto acc_old) {
        auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

        if constexpr(i.value < N - 1)
        {
            return fs(fs, i + Number<1>{}, acc_new);
        }
        else
        {
            return acc_new;
        }
    };**/

    // const auto element_space_size = f(f, Number<0>{}, LongNumber<1>{});
    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                       element_space_size};
}

template <typename NewTransforms>
struct lambda_get_up_dim_num
{
    template <typename I>
    constexpr auto operator()(I) const
    {
        using Tran = remove_reference_t<decltype(NewTransforms{}.At(I{}))>;
        return Number<Tran::GetNumOfUpperDimension()>{};
    }
};

template <typename OldTensorDescriptor,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
constexpr auto transform_tensor_descriptor(const OldTensorDescriptor& old_tensor_desc,
                                           const NewTransforms& new_transforms,
                                           NewLowerDimensionOldVisibleIdss,
                                           NewUpperDimensionNewVisibleIdss)
{
    {
        static_assert(NewTransforms::Size() == NewLowerDimensionOldVisibleIdss::Size() &&
                          NewTransforms::Size() == NewUpperDimensionNewVisibleIdss::Size(),
                      "wrong! inconsitent number of transform");

        constexpr auto all_old_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewLowerDimensionOldVisibleIdss{});

        constexpr auto all_new_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewUpperDimensionNewVisibleIdss{});

        static_assert(is_valid_sequence_map<decltype(all_old_top_ids)>::value &&
                          is_valid_sequence_map<decltype(all_new_top_ids)>::value,
                      "wrong!");
    }
    constexpr auto low_dim_hidden_idss = transform_tuples(
        [](auto low_dim_visible_ids) constexpr {
            return transform_sequences(
                [](auto low_dim_visible_id) constexpr {
                    return OldTensorDescriptor::GetVisibleDimensionIds()[low_dim_visible_id];
                },
                low_dim_visible_ids);
        },
        NewLowerDimensionOldVisibleIdss{});

    constexpr index_t num_new_transform     = NewTransforms::Size();
    constexpr index_t old_hidden_dim_number = OldTensorDescriptor::GetNumOfHiddenDimension();

    constexpr auto up_dim_numbers =
        generate_sequence(lambda_get_up_dim_num<NewTransforms>{}, Number<num_new_transform>{});

    constexpr auto up_dim_numbers_scan = merge_sequences(
        Sequence<0>{}, inclusive_scan_sequence(up_dim_numbers, plus<index_t>{}, Number<0>{}));

    constexpr auto up_dim_hidden_idss = generate_tuple(
        [ old_hidden_dim_number, up_dim_numbers_scan ](auto i) constexpr {
            return
                typename arithmetic_sequence_gen<old_hidden_dim_number + up_dim_numbers_scan[i],
                                                 old_hidden_dim_number + up_dim_numbers_scan[i + 1],
                                                 1>::type{};
        },
        Number<num_new_transform>{});

    constexpr auto unordered_new_visible_dim_hidden_ids = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); }, up_dim_hidden_idss);

    constexpr auto new_visible_dim_unordered2ordered = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); },
        NewUpperDimensionNewVisibleIdss{});

    constexpr auto new_visible_dim_hidden_ids =
        unordered_new_visible_dim_hidden_ids.ReorderGivenOld2New(new_visible_dim_unordered2ordered);
    const auto all_transforms = container_concat(old_tensor_desc.GetTransforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        container_concat(OldTensorDescriptor::GetLowerDimensionIdss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        container_concat(OldTensorDescriptor::GetUpperDimensionIdss(), up_dim_hidden_idss);

    const auto element_space_size = old_tensor_desc.GetElementSpaceSize();

    return TensorDescriptor<remove_cv_t<decltype(all_transforms)>,
                            remove_cv_t<decltype(all_low_dim_hidden_idss)>,
                            remove_cv_t<decltype(all_up_dim_hidden_idss)>,
                            remove_cv_t<decltype(new_visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{all_transforms,
                                                                       element_space_size};
}

template <typename LowLength>
struct Pass
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    constexpr Pass() = default;

    constexpr Pass(const LowLength& low_length) : up_lengths_{make_tuple(low_length)} {}

    static constexpr index_t GetNumOfLowerDimension() { return 1; }

    static constexpr index_t GetNumOfUpperDimension() { return 1; }

    constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    static constexpr void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }
    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                 const UpIdxDiff& idx_diff_up,
                                 LowIdx& idx_low,
                                 const UpIdx&,
                                 Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    static constexpr bool IsLinearTransform() { return true; }

    static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex() { return true; }

    template <typename UpIdx>
    static constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }
};

template <typename LowLength>
constexpr auto make_pass_through_transform(const LowLength& low_length)
{
    return Pass<LowLength>{low_length};
}

} // namespace host
} // namespace ck
