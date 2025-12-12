// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <memory>
#include <numeric>
#include <span>
#include <concepts>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/host/host_tensor.hpp"

/// This file deals with tensor memory allocation: Both the act of allocating
/// and (automatically) deallocating memory, as well as utilities for managing
/// the layout of tensor data in memory.

namespace ck_tile::builder::test {

/// @brief Automatic deleter for GPU memory.
///
/// This structure implements a C++ functor which can be used to configure
/// `std::unique_ptr` to automatically delete memory using `hipFree`.
///
/// @see DeviceBuffer
struct DeviceMemoryDeleter
{
    /// @brief Deleter callback.
    ///
    /// This function is invoked by `std::unique_ptr` when memory that the
    /// pointer represents should be freed. In our implementation, we just
    /// pass it directly to `hipFree`.
    void operator()(std::byte* ptr) const
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

/// @brief HIP out of memory error
///
/// This is a derivation of `std::runtime_error` specialized for HIP
/// out-of-memory errors.
///
/// @see std::runtime_error
struct OutOfDeviceMemoryError : std::runtime_error
{
    /// @brief Utility for formatting out-of-memory error messages
    ///
    /// Returns a human-readable description of a HIP out-of-memory error.
    ///
    /// @param status The status to report
    static std::string format_error(hipError_t status)
    {
        return std::string("failed to allocate hip memory: ") + hipGetErrorString(status) + " (" +
               std::to_string(status) + ")";
    }

    /// @brief Construct an out-of-memory error using `status` as message.
    ///
    /// @param status A HIP error status that was encountered while allocating memory.
    OutOfDeviceMemoryError(hipError_t status) : std::runtime_error(format_error(status)) {}
};

/// @brief Automatically managed GPU memory.
///
/// The `DeviceBuffer` is an automatically managed pointer for GPU memory. When
/// adopting a device pointer into a `DeviceBuffer`, it will automatically be
/// free'd when the pointer goes out of scope. Memory can be allocated directly
/// into a `DeviceBuffer` using `alloc_buffer()` or `alloc_tensor_buffer()`.
///
/// Since this type is just an alias of `std::unique_ptr`, you can use that type's
/// functionality to manage memory further, such as `.reset()` to release the
/// memory.
///
/// @see alloc_buffer()
/// @see alloc_tensor_buffer()
using DeviceBuffer = std::unique_ptr<std::byte[], DeviceMemoryDeleter>;

/// @brief Allocate automatically managed GPU memory.
///
/// This function essentially acts like a managed version of hipMalloc -
/// allocating GPU memory on the currently active device - except that this
/// version returns an automatically managed pointer.
///
/// @param size The amount of memory to allocate in bytes.
/// @throws OutOfDeviceMemoryError if memory allocation failed.
///
/// @see DeviceBuffer
/// @see OutOfDeviceMemoryError
/// @see hipMalloc()
inline DeviceBuffer alloc_buffer(size_t size)
{
    std::byte* d_buf = nullptr;
    if(const auto status = hipMalloc(&d_buf, size); status != hipSuccess)
    {
        throw OutOfDeviceMemoryError(status);
    }
    return DeviceBuffer(d_buf);
}

/// @brief Type managing tensor data layout in memory.
///
/// This structure describes a tensor in memory. It does not actually hold any
/// reference to memory, it just describes how the memory should be laid out if it
/// were.
///
/// @note This type is very much like ck_tile::HostTensorDescriptor, except that it
/// also  includes the data type of the elements of htis tensor. This is mainly to
/// make the descriptor a _complete_ description of a tensor rather than just the
/// dimensions in strides, which helps in reducing clutter in uses of this type.
///
/// @note All strides are still in _elements_.
///
/// @tparam DT The conceptual data type of the tensor elements. This need not be the
///   type that the data is actually stored as in memory.
template <DataType DT>
struct TensorDescriptor
{
    // For now, the implementation of this type is based on
    // `ck_tile::HostTensorDescriptor`, so that we can prototype without
    // reimplementing the `HostTensorDescriptor` for the 3rd time. You can regard
    // the use of `ck_tile::HostTensorDescriptor` here as an implementation detail.

    /// The conceptual data type of the tensor elements. This need not be the type
    /// that the data is actually stored as in memory.
    constexpr static DataType data_type = DT;

    /// @brief Create a tensor descriptor from lengths and strides.
    ///
    /// @param lengths A sequence of tensor lengths, the conceptial dimensions of
    ///   the tensor in  elements.
    /// @param strides A sequence of in-memory strides of the tensor, measured in
    ///   elements. Each element of `strides`` corresponds to one at the same index
    ///   in `lengths`, the amount of elements to skip in memory to find the next
    ///   element along that axis.
    TensorDescriptor(std::span<const size_t> lengths, std::span<const size_t> strides)
        : inner_descriptor_(lengths, strides)
    {
        // TODO: Validation of strides? For now we just delegate the details of the
        // construction to the CK Tile HostTensorDescriptor.
    }

    /// Query the conceptual dimensions of the tensor.
    ///
    /// @returns A span of tensor dimensions, one for every axis. Note that the order
    ///   does *not* correspond with memory layout, query the in-memory strides for
    ///   that.
    ///
    /// @see get_strides()
    std::span<const size_t> get_lengths() const { return inner_descriptor_.get_lengths(); }

    /// Query the in-memory strides of the tensor.
    ///
    /// @returns A span of tensor dimensions, one for every axis. Each element
    ///   corresponds directly with the stride in elements at the same index in the
    ///   tensor  dimensions.
    ///
    /// @see get_lengths()
    std::span<const size_t> get_strides() const { return inner_descriptor_.get_strides(); }

    /// @brief Compute total tensor size in elements.
    ///
    /// This function returns the total size of the memory backing a tensor with
    /// this descriptor in *elements*, including required extra size for strides.
    ///
    /// @see get_element_space_size_in_bytes()
    size_t get_element_space_size() const { return inner_descriptor_.get_element_space_size(); }

    /// @brief Compute total tensor size in bytes.
    ///
    /// This function is like `get_element_space_size()`, except that the returned
    /// value is measured in *bytes* rather than *elements*. Use this function for
    /// figuring out how much memory needs to be allocated for a particular tensor.
    ///
    /// @see get_element_space_size()
    size_t get_element_space_size_in_bytes() const
    {
        // For now, the backing type is the naive C++-type that represents the data
        // type. When we are going to support packed types such as i4 and fp6, this
        // is going to become more complicated.
        return get_element_space_size() * data_type_sizeof(DT);
    }

    private:
    ck_tile::HostTensorDescriptor inner_descriptor_;
};

/// @brief Allocate automatically managed GPU memory corresponding to a tensor descriptor.
///
/// This function is similar to `alloc_buffer()`, except that the required size is
/// derived automatically from a tensor descriptor. The returned buffer is valid for
/// tensors with that layout. Strides are also taken into account when computing the
/// required size.
///
/// @tparam DT The conceptual datatype of the elements of the tensor.
/// @param descriptor A descriptor of the memory layout of the tensor to allocate.
/// @throws OutOfDeviceMemoryError if memory allocation failed.
///
/// @see TensorDescriptor
/// @see DeviceBuffer
/// @see OutOfDeviceMemoryError
/// @see hipMalloc()
template <DataType DT>
DeviceBuffer alloc_tensor_buffer(const TensorDescriptor<DT>& descriptor)
{
    return alloc_buffer(descriptor.get_element_space_size_in_bytes());
}

} // namespace ck_tile::builder::test
