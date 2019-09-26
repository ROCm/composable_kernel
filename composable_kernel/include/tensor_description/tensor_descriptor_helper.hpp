#ifndef CK_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

template <typename Lengths>
__host__ __device__ constexpr auto calculate_tensor_strides_packed(Lengths)
{
    return reverse_inclusive_scan_sequence(
               Lengths{}.PopFront(), math::multiplies<index_t>{}, Number<1>{})
        .PushBack(Number<1>{});
}

template <typename Lengths, index_t Align>
__host__ __device__ constexpr auto calculate_tensor_strides_aligned(Lengths, Number<Align>)
{
    constexpr index_t L_back_align =
        Align * math::integer_divide_ceiler<index_t>{}(Lengths{}.Back(), Align);

    return calculate_tensor_strides_packed(
        Lengths{}.Modify(Number<Lengths{}.GetSize() - 1>{}, Number<L_back_align>{}));
}

template <index_t... Lengths, index_t... Strides>
__host__ __device__ constexpr auto make_native_tensor_descriptor(Sequence<Lengths...>,
                                                                 Sequence<Strides...>)
{
    return NativeTensorDescriptor<NativeDimension<Lengths, Strides>...>{};
}

template <typename Lengths>
__host__ __device__ constexpr auto make_native_tensor_descriptor_packed(Lengths)
{
    constexpr auto strides = calculate_tensor_strides_packed(Lengths{});

    return make_native_tensor_descriptor(Lengths{}, strides);
}

template <typename Lengths, index_t Align>
__host__ __device__ constexpr auto make_native_tensor_descriptor_aligned(Lengths, Number<Align>)
{
    constexpr auto strides = calculate_tensor_strides_aligned(Lengths{}, Number<Align>{});
    return make_native_tensor_descriptor(Lengths{}, strides);
}

template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensionIds,
          typename UpDimensionIds>
__host__ __device__ constexpr auto
    transform_tensor_descriptor(LowTensorDescriptor, Transforms, LowDimensionIds, UpDimensionIds)
{
    return TransformedTensorDescriptor<LowTensorDescriptor,
                                       Transforms,
                                       LowDimensionIds,
                                       UpDimensionIds>{};
}

template <typename LowerTensorDescriptor,
          index_t... LowerLengths,
          index_t... LowerDimensionIds,
          index_t... UpperDimensionIds>
__host__ __device__ constexpr auto reorder_tensor_descriptor_impl(LowerTensorDescriptor,
                                                                  Sequence<LowerLengths...>,
                                                                  Sequence<LowerDimensionIds...>,
                                                                  Sequence<UpperDimensionIds...>)
{
    return TransformedTensorDescriptor<LowerTensorDescriptor,
                                       Tuple<PassThrough<LowerLengths>...>,
                                       Tuple<Sequence<LowerDimensionIds>...>,
                                       Tuple<Sequence<UpperDimensionIds>...>>{};
}

template <typename LowerTensorDescriptor, typename MapLower2Upper>
__host__ __device__ constexpr auto
    reorder_tensor_descriptor_given_lower2upper(LowerTensorDescriptor, MapLower2Upper)
{
    static_assert(is_valid_sequence_map<MapLower2Upper>{},
                  "wrong! MapLower2Upper is not a valid map");

    return reorder_tensor_descriptor_impl(
        LowerTensorDescriptor{},
        LowerTensorDescriptor::GetLengths(),
        typename arithmetic_sequence_gen<0, LowerTensorDescriptor::GetNumOfDimension(), 1>::type{},
        MapLower2Upper{});
}

template <typename LowerTensorDescriptor, typename MapUpper2Lower>
__host__ __device__ constexpr auto
    reorder_tensor_descriptor_given_upper2lower(LowerTensorDescriptor, MapUpper2Lower)
{
    return reorder_tensor_descriptor_given_lower2upper(
        LowerTensorDescriptor{}, typename sequence_map_inverse<MapUpper2Lower>::type{});
}

template <typename Lengths, typename Strides>
__host__ __device__ constexpr bool AreDimensionsUnfoldable(Lengths, Strides)
{
    static_assert(Lengths::Size() == Strides::Size(), "wrong!");

    bool flag = true;

    for(index_t i = 0; i < Lengths::Size() - 1; ++i)
    {
        flag = flag && Strides::At(i) == Strides::At(i + 1) * Lengths::At(i + 1);
    }

    return flag;
}

// unfold only support NativeTennsorDescriptor, for now
template <index_t FirstUnfoldDim, index_t LastUnfoldDim, typename... Ts>
__host__ __device__ constexpr auto unfold_tensor_descriptor(NativeTensorDescriptor<Ts...> desc,
                                                            Number<FirstUnfoldDim>,
                                                            Number<LastUnfoldDim>)
{
    constexpr index_t nDim = desc.GetNumOfDimension();

    static_assert(FirstUnfoldDim >= 0 && LastUnfoldDim < nDim && FirstUnfoldDim <= LastUnfoldDim,
                  "wrong! should have FirstUnfoldDim <= LastUnfoldDim!");

    // left and right
    constexpr auto left = typename arithmetic_sequence_gen<0, FirstUnfoldDim, 1>::type{};
    constexpr auto middle =
        typename arithmetic_sequence_gen<FirstUnfoldDim, LastUnfoldDim + 1, 1>::type{};
    constexpr auto right = typename arithmetic_sequence_gen<LastUnfoldDim + 1, nDim, 1>::type{};

    // sanity-checknfoldable
    static_assert(AreDimensionsUnfoldable(desc.GetLengths(middle), desc.GetStrides(middle)),
                  "wrong! not unfoldable");

    // unfolded length, stride
    constexpr index_t unfold_length =
        reduce_on_sequence(desc.GetLengths(middle), math::multiplies<index_t>{}, Number<1>{});

    constexpr index_t unfold_stride = desc.GetStride(Number<LastUnfoldDim>{});

    // new lengths, strides
    constexpr auto new_lengths =
        desc.GetLengths(left).PushBack(Number<unfold_length>{}).PushBack(desc.GetLengths(right));

    constexpr auto new_strides =
        desc.GetStrides(left).PushBack(Number<unfold_stride>{}).PushBack(desc.GetStrides(right));

    return make_native_tensor_descriptor(new_lengths, new_strides);
}

#if 0
// not implemented
template <typename LowerTensorDescriptor,
          typename PadDimensionIds,
          typename LeftPads,
          typename RightPads>
__host__ __device__ constexpr auto
    pad_tensor_descriptor(LowerTensorDescriptor, PadLowerDimensionIds, LeftPads, RightPads)
{
    constexpr index_t nDim = LowerTensorDescriptor::GetNumOfDimension();

    constexpr auto non_pad_low_dim_ids = xxx;

    return transform_tensor_descriptor(
        LowerTensorDescriptor{},
        make_tuple(Pad<decltype(LowerTensorDescriptor::GetLengths(PadLowerDimensionIds{})),
                       LeftPads,
                       RightPads>{})
            .PushBack(PassThrough<xxxx>...),
        make_tuple(PadLowerDimensionIds{}).PushBack(xxxx),
        sequence_to_tuple(typename arithmetic_sequence_gen<0, nDim, 1> i::type{}));
}
#endif

// a cluster map 1d index to N-d index
template <typename Lengths, typename ArrangeOrder>
struct ClusterDescriptor
{
    static constexpr index_t nDim = Lengths::Size();

    static constexpr auto mDesc = transform_tensor_descriptor(
        make_native_tensor_descriptor_packed(Lengths{}),
        make_tuple(Merge<decltype(Lengths::ReorderGivenNew2Old(ArrangeOrder{}))>{}),
        make_tuple(ArrangeOrder{}),
        make_tuple(Sequence<0>{}));

    __host__ __device__ constexpr ClusterDescriptor()
    {
        static_assert(Lengths::Size() == nDim && ArrangeOrder::Size() == nDim,
                      "wrong! size not the same");

        static_assert(is_valid_sequence_map<ArrangeOrder>{}, "wrong! ArrangeOrder is wrong");
    }

    __host__ __device__ static constexpr index_t GetElementSize() { return mDesc.GetElementSize(); }

    __host__ __device__ static constexpr auto CalculateClusterIndex(index_t idx_1d)
    {
        return mDesc.CalculateLowerIndex(MultiIndex<1>{idx_1d});
    }
};

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor(
    Lengths, ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    return ClusterDescriptor<Lengths, ArrangeOrder>{};
}

template <typename... NativeDimensions>
__host__ __device__ void
print_tensor_descriptor(const char* s, const NativeTensorDescriptor<NativeDimensions...>& desc)
{
    print_tensor_descriptor_impl(s, desc.GetLengths(), desc.GetStrides());
}

template <typename... Ts>
__host__ __device__ void print_tensor_descriptor(const char* s,
                                                 const TransformedTensorDescriptor<Ts...>& desc)
{
    print_tensor_descriptor_impl(s, desc.GetLengths());
}

template <index_t... Lengths, index_t... Strides>
__host__ __device__ void
print_tensor_descriptor_impl(const char* s, Sequence<Lengths...>, Sequence<Strides...>)
{
    constexpr index_t nDim = sizeof...(Lengths);

    static_assert(nDim > 0 && nDim <= 12, "wrong!");

    static_if<nDim == 1>{}([&](auto) {
        printf("%s dim %u, lengths {%u}, strides {%u}\n", s, nDim, Lengths..., Strides...);
    });

    static_if<nDim == 2>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u}, strides {%u %u}\n", s, nDim, Lengths..., Strides...);
    });

    static_if<nDim == 3>{}([&](auto) {
        printf(
            "%s dim %u, lengths {%u %u %u}, strides {%u %u %u}\n", s, nDim, Lengths..., Strides...);
    });

    static_if<nDim == 4>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u}, strides {%u %u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 5>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u}, strides {%u %u %u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 6>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u}, strides {%u %u %u %u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 7>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 8>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 9>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u %u "
               "%u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 10>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u %u "
               "%u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 11>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u %u "
               "%u %u "
               "%u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });

    static_if<nDim == 12>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u %u %u}, strides {%u %u %u %u %u "
               "%u %u %u %u "
               "%u %u %u}\n",
               s,
               nDim,
               Lengths...,
               Strides...);
    });
}

template <index_t... Lengths>
__host__ __device__ void print_tensor_descriptor_impl(const char* s, Sequence<Lengths...>)
{
    constexpr index_t nDim = sizeof...(Lengths);

    static_assert(nDim > 0 && nDim <= 12, "wrong!");

    static_if<nDim == 1>{}([&](auto) { printf("%s dim %u, lengths {%u}\n", s, nDim, Lengths...); });

    static_if<nDim == 2>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u}\n", s, nDim, Lengths...); });

    static_if<nDim == 3>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u %u}\n", s, nDim, Lengths...); });

    static_if<nDim == 4>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u %u %u}\n", s, nDim, Lengths...); });

    static_if<nDim == 5>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u %u %u %u}\n", s, nDim, Lengths...); });

    static_if<nDim == 6>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u %u %u %u %u}, \n", s, nDim, Lengths...); });

    static_if<nDim == 7>{}(
        [&](auto) { printf("%s dim %u, lengths {%u %u %u %u %u %u %u}\n", s, nDim, Lengths...); });

    static_if<nDim == 8>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u}\n", s, nDim, Lengths...);
    });

    static_if<nDim == 9>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u}\n", s, nDim, Lengths...);
    });

    static_if<nDim == 10>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u}\n", s, nDim, Lengths...);
    });

    static_if<nDim == 11>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u %u}\n", s, nDim, Lengths...);
    });

    static_if<nDim == 12>{}([&](auto) {
        printf("%s dim %u, lengths {%u %u %u %u %u %u %u %u %u %u %u %u}\n", s, nDim, Lengths...);
    });
}

} // namespace ck
#endif
