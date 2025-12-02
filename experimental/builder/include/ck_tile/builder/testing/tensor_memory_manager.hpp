// Copyright (c) advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <numeric>
#include <span>
#include <concepts>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile::builder::test {

struct DeviceMemoryDeleter
{
    void operator()(std::byte* ptr) const
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

using DeviceBuffer = std::unique_ptr<std::byte[], DeviceMemoryDeleter>;

inline DeviceBuffer alloc_buffer(size_t size)
{
    std::byte* d_buf  = nullptr;
    const auto status = hipMalloc(&d_buf, size);
    // TODO(Robin): How to check error without relying on google test?
    // Ideally we get some sort of trace here, but thats not possible until c++23.
    // For now just throw a runtime error.
    if(status != hipSuccess)
    {
        throw std::runtime_error("failed to allocate hip memory");
    }
    return DeviceBuffer(d_buf);
}

/// This structure describes a tensor in memory. It does not actually hold any reference
/// to memory, it just describes how the memory should be laid out if it were.
///
/// This type is very much like ck_tile::HostTensorDescriptor, except that it also
/// includes the data type of the elements of htis tensor. This is mainly to
/// make the descriptor a _complete_ description of a tensor rather than just the
/// dimensions in strides, which helps in reducing clutter in uses of this type.
/// Note that all strides are still in _elements_.
template <DataType DT>
struct TensorDescriptor
{
    constexpr static DataType data_type = DT;

    // For now, the implementation of this type is based on `ck_tile::HostTensorDescriptor`,
    // so that we can prototype without reimplementing the `HostTensorDescriptor` for the
    // 3rd time. You can regard the use of `ck_tile::HostTensorDescriptor` here as an
    // implementation detail.

    /// Main constructor for a `HostTensorDescriptor`.
    /// - `lengths` is a set of tensor lengths, the conceptial dimensions of the tensor in
    ///   elements.
    /// - `strides` are the in-memory strides of the tensor, measured in elements. Each
    ///   element of `strides`` corresponds to one at the same index in `lengths`, the
    ///   amount of elements to skip in memory to find the next element along that axis.
    TensorDescriptor(std::span<const size_t> lengths, std::span<const size_t> strides)
        : inner_descriptor_(lengths, strides)
    {
        // TODO: Validation of strides? For now we just delegate the details of the construction to
        // the CK Tile HostTensorDescriptor.
    }

    std::span<const size_t> get_lengths() const { return inner_descriptor_.get_lengths(); }
    std::span<const size_t> get_strides() const { return inner_descriptor_.get_strides(); }

    /// This function returns the total size of the memory backing a tensor with this
    /// descriptor in *elements*, including required extra size for strides.
    size_t get_element_space_size() const { return inner_descriptor_.get_element_space_size(); }

    /// This function is like `get_element_space_size()`, except that the returned value is
    /// measured in *bytes* rather than *elements*. Use this function for figuring out how
    /// much memory needs to be allocated for a particular tensor.
    size_t get_element_space_size_in_bytes() const
    {
        // For now, the backing type is the naive C++-type that represents the data type.
        // When we are going to support packed types such as i4 and fp6, this is going to
        // become more complicated.
        return get_element_space_size() * sizeof(typename DataTypeTraits<DT>::Type);
    }

    private:
    ck_tile::HostTensorDescriptor inner_descriptor_;
};

template <DataType DT>
DeviceBuffer alloc_tensor_buffer(const TensorDescriptor<DT>& descriptor)
{
    return alloc_buffer(descriptor.get_element_space_size_in_bytes());
}

} // namespace ck_tile::builder::test
