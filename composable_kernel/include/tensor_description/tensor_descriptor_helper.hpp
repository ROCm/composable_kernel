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

template <typename LowerTensorDescriptor, index_t VectorDim, index_t VectorSize>
__host__ __device__ constexpr auto
vectorize_tensor_descriptor(LowerTensorDescriptor, Number<VectorDim> vector_dim, Number<VectorSize>)
{
    constexpr index_t nDim = LowerTensorDescriptor::GetNumOfDimension();

    return transform_tensor_descriptor(
        LowerTensorDescriptor{},
        Vectorize<LowerTensorDescriptor::GetLength(vector_dim), VectorSize>{},
        typename arithmetic_sequence_gen<0, nDim, 1>::type{},
        typename arithmetic_sequence_gen<0, nDim, 1>::type{});
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
