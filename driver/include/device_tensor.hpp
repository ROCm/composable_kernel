#pragma once
#include "tensor.hpp"
#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "tensor_descriptor.hpp"

template <typename ConstTensorDesc, std::size_t... Is>
auto make_TensorDescriptor_impl(ConstTensorDesc, std::integer_sequence<std::size_t, Is...>)
{
    std::initializer_list<std::size_t> lengths = {ConstTensorDesc::GetLengths()[Is]...};
    std::initializer_list<std::size_t> strides = {ConstTensorDesc::GetStrides()[Is]...};

    return TensorDescriptor(lengths, strides);
}

template <typename ConstTensorDesc>
auto make_TensorDescriptor(ConstTensorDesc)
{
    return make_TensorDescriptor_impl(
        ConstTensorDesc{},
        std::make_integer_sequence<std::size_t, ConstTensorDesc::GetNumOfDimension()>{});
}

template <typename ConstTensorDesc>
void ostream_ConstantTensorDescriptor(ConstTensorDesc, std::ostream& os = std::cout)
{
    ostream_TensorDescriptor(make_TensorDescriptor(ConstTensorDesc{}), os);
}
