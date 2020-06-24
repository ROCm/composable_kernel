#pragma once
#include "host_tensor.hpp"
#include "common_header.hpp"
#include "tensor_descriptor.hpp"

template <typename TensorDesc, std::size_t... Is>
auto make_HostTensorDescriptor_impl(TensorDesc, std::integer_sequence<std::size_t, Is...>)
{
    std::initializer_list<std::size_t> lengths = {TensorDesc::GetLengths()[Is]...};
    std::initializer_list<std::size_t> strides = {TensorDesc::GetStrides()[Is]...};

    return HostTensorDescriptor(lengths, strides);
}

template <typename TensorDesc>
auto make_HostTensorDescriptor(TensorDesc)
{
    return make_HostTensorDescriptor_impl(
        TensorDesc{}, std::make_integer_sequence<std::size_t, TensorDesc::GetNumOfDimension()>{});
}

template <typename TensorDesc>
void ostream_tensor_descriptor(TensorDesc, std::ostream& os = std::cout)
{
    ostream_HostTensorDescriptor(make_HostTensorDescriptor(TensorDesc{}), os);
}
