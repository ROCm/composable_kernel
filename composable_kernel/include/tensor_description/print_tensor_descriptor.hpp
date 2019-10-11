#ifndef CK_PRINT_TENSOR_DESCRIPTOR_HPP
#define CK_PRINT_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

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
