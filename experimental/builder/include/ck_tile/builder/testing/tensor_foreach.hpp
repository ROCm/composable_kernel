// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include <cstdint>
#include <cassert>
#include <concepts>
#include <array>

/// This file implements a generic GPU tensor "foreach" function. This
/// functionality turned out useful in separate parts of the testing
/// system, hence its implemented in a separate file. This version is
/// not particularly efficient (but it should at least be readable),
/// but it should be easy to replace the implementation in the future,
/// should that be needed.

namespace ck_tile::builder::test {

/// @brief Utility structure for N-dimensional iteration using a flat index
///
/// This structure's main purpose is to "unmerge" a flattened index into a
/// multi-dimensional index, which helps when iterating over multi-dimensional
/// indices without having to write an arbitrary amount of nested for loops.
/// A minimal amount of precomputation must be done to do this efficiently,
/// which is handled in the constructor of this type.
///
/// @details Decoding a flat index into a multi-dimensional index is done by
/// first computing a reverse scan of the shape. These values can then be
/// used to decode the index in the usual way:
///
///     x = flat_idx / (size_y * size_z)
///     y = flat_idx % (size_y * size_z) / size_z
///     z = flat_idx % (size_y * size_z) % size_z
///     etc
///
/// The decode order is such that the innermost dimension (right in
/// the shape extent) changes the fastest.
///
/// @tparam RANK The rank (number of spatial dimensions) of the tensor to
/// iterate.
template <size_t RANK>
struct NdIter
{
    /// @brief Prepare N-dimensional iteration over a particular shape.
    ///
    /// Precompute ashape into a form that can be used to easily decode a flat
    /// index into a multi-dimensional index.
    ///
    /// @param shape The shape to iterate over.
    explicit NdIter(const Extent<RANK>& shape)
    {
        // Precompute shape_scan = [..., shape[-2] * shape[-1], shape[-1], 1]

        numel_ = 1;
        for(int i = RANK; i > 0; --i)
        {
            shape_scan_[i - 1] = numel_;
            numel_ *= shape[i - 1];
        }
    }

    /// @brief Unflatten a flat index into a multi-dimensional index
    ///
    /// This applies the usual multi-dimensional indexing method over the
    /// precomputed shape scan to get back a multi-dimensional index.
    /// The decode order is such that the innermost dimension (right in
    /// the shape extent) changes the fastest.
    ///
    /// @param flat_index The "flattened" (1-dimensional) index of the tensor
    ///
    /// @returns A multi-dimensional index into the tensor
    ///
    /// @pre `0 <= flat_index < size()` (in other words, the `flat_index` must
    /// be in bounds of the tensor shape that this `NdIter` was made from).
    __host__ __device__ Extent<RANK> operator()(size_t flat_index) const
    {
        Extent<RANK> index = {};
        auto idx           = flat_index;
        for(size_t i = 0; i < RANK; ++i)
        {
            const auto scanned_dim = shape_scan_[i];
            index[i]               = idx / scanned_dim;
            idx %= scanned_dim;
        }

        return index;
    }

    /// @brief Return the total elements to iterate over
    ///
    /// Get the total number of elements in the shape to iterate over. This value
    /// can be used to construct a complete for loop to iterate over all indices
    /// of a tensor, for example:
    ///
    ///    for(size_t i = 0; i < iter.numel(); ++i)
    ///    {
    ///        const auto index = iter(i);
    ///        use(index);
    ///    }
    __host__ __device__ size_t numel() const { return numel_; }

    private:
    /// Reverse (right) scan of the shape to iterate over.
    Extent<RANK> shape_scan_;

    /// The total number of elements in the shape. This value turns out to be almost
    /// always required when iterating over a shape, so just store it in this type
    /// so that it is easily accessible.
    size_t numel_;
};

template <size_t RANK>
NdIter(Extent<RANK>) -> NdIter<RANK>;

/// @brief Concept for constraining tensor iteration functors.
///
/// This concept checks that a functor has the correct signature for
/// use with the `tensor_foreach` function.
template <typename F, int RANK>
concept ForeachFunctor = requires(const F& f, const Extent<RANK>& index) {
    { f(index) } -> std::same_as<void>;
};

namespace detail {

/// @brief Default foreach kernel block size
///
/// This value is the default number of threads in each block when
/// executing the foreach kernel. This value is mostly arbitrary,
/// 256 is usually a good default for AMD GPUs.
///
/// @see tensor_foreach
constexpr int DEVICE_FOREACH_BLOCK_SIZE = 256;

/// @brief Tensor iteration kernel
///
/// This kernel implements the actual iteration logic, and is intended
/// to be used solely by `tensor_foreach` to iterate & invoke the
/// actual callback.
///
/// @tparam BLOCK_SIZE The number of threads in each block on the GPU.
/// @tparam RANK The rank (number of spatial dimensions) of the tensor to
/// iterate.
/// @tparam F The type of the callback to invoke. This function must be
/// compatible with execution as a __device__ function.
///
/// @param iter An NdIter instance to help iterating over the tensor.
/// @param f The callback to invoke for each index of the tensor. This
/// functor must be eligible for running on the GPU.
template <int BLOCK_SIZE, size_t RANK, typename F>
    requires ForeachFunctor<F, RANK>
__global__ __launch_bounds__(BLOCK_SIZE) //
    void foreach_kernel(NdIter<RANK> iter, F f)
{
    const auto gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(size_t flat_idx = gid; flat_idx < iter.numel(); flat_idx += gridDim.x * BLOCK_SIZE)
    {
        // Compute the current index.
        const auto index = iter(flat_idx);

        // Then invoke the callback with the index.
        f(index);
    }
}

/// @brief A utility to get a C++ type for a CKB type
///
/// Right now this is just an alias of an internal CKB helper,
/// but this should probably be moved elsewhere.
template <builder::DataType DT>
using cpp_type_t = typename builder::factory::internal::DataTypeToCK<DT>::type;

} // namespace detail

/// @brief Calculate tensor memory offset given index and strides.
///
/// This function returns the offset in memory in a tensor, given a particular
/// multi-dimensional index and a particular set of strides. Each value in the
/// index corresponds one-to-one with a value in the strides, which are the
/// index and stride at that dimension in the tensor. These strides must be
/// pre-scanned, meaning that each index is the absolute stride of elements
/// along that axis. In essence, this means that you should pass the output of
/// `TensorDescriptor::get_strides()` into this function.
///
/// @pre The index must be inside the tensor space.
///
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param index A multi-dimensional index inside the tensor space.
/// @param strides A set of strides, one for each dimension.
///
/// @see TensorDescriptor
template <size_t RANK>
__host__ __device__ size_t calculate_offset(const Extent<RANK>& index, const Extent<RANK>& strides)
{
    size_t offset = 0;
#pragma unroll
    for(size_t i = 0; i < RANK; ++i)
    {
        offset += index[i] * strides[i];
    }
    return offset;
}

/// @brief Invoke a callback on the GPU for every index in a tensor.
///
/// This function invokes a callback functor on the GPU, for each index in
/// a tensor. This function _only_ takes care of iterating over all indices
/// in a tensor of a particular shape; this function does not handle or know
/// about actual tensor data.
///
/// @note This function is currently implemented relatively naively: The
/// iteration order is always row-wise, implemented as a persistent kernel.
/// The main objective of this function is to be used with the CK-Builder
/// testing system, and so readability and correctness should be preferred
/// over performance. If this is ever a source of performance problems,
/// feel free to replace the implementation with something better.
///
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param shape The shape of the tensor to iterate over.
/// @param f The callback to invoke for each index of the tensor. This
/// functor must be eligible for running on the GPU.
///
/// @see ForeachFunctor
/// @see detail::foreach_kernel
template <size_t RANK>
void tensor_foreach(const Extent<RANK>& shape, ForeachFunctor<RANK> auto f)
{
    constexpr int block_size = detail::DEVICE_FOREACH_BLOCK_SIZE;
    const auto kernel        = detail::foreach_kernel<block_size, RANK, decltype(f)>;

    int occupancy;
    check_hip(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    int device;
    check_hip(hipGetDevice(&device));

    int multiprocessors;
    check_hip(
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device));

    // Pre-scan the shape to help indexing in the kernel.
    // Note: the order is not that important, so long as the iteration
    // order in the kernel is from large-to-small. Right layout is the
    // easiest solution for that.

    NdIter iter(shape);

    // Reset any errors from previous launches.
    (void)hipGetLastError();

    kernel<<<occupancy * multiprocessors, block_size>>>(iter, f);
    check_hip(hipGetLastError());
}

/// @brief Concept for tensor initializing functors.
///
/// This concept checks that a functor has the correct signature for
/// use with the `fill_tensor` function.
template <typename F, DataType DT, size_t RANK>
concept FillTensorFunctor = requires(const F& f, const Extent<RANK>& index) {
    { f(index) } -> std::convertible_to<detail::cpp_type_t<DT>>;
};

/// @brief Utility for initializing tensors.
///
/// This function is a utility helper for initializing tensors. It accepts a
/// tensor descriptor, buffer, and a callback. The callback is invoked for every
/// coordinate (which is passed to the callback), and the tensor is initialized
/// with resulting value.
///
/// @tparam DT The tensor element datatype
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param desc The descriptor of the tensor to initialize.
/// @param buffer The memory of the tensor to initialize.
/// @param f A functor used to get the value at a particular coordinate.
///
/// @see FillTensorFunctor
template <DataType DT, size_t RANK>
void fill_tensor(const TensorDescriptor<DT, RANK>& desc,
                 void* buffer,
                 FillTensorFunctor<DT, RANK> auto f)
{
    const auto strides = desc.get_strides();
    tensor_foreach(desc.get_lengths(), [buffer, f, strides](const auto& index) {
        using T           = detail::cpp_type_t<DT>;
        auto* ptr         = static_cast<T*>(buffer);
        const auto offset = calculate_offset(index, strides);

        ptr[offset] = f(index);
    });
}

/// @brief Concept for tensor buffer initializing functors.
///
/// This concept checks that a functor has the correct signature for
/// use with the `fill_tensor_buffer` function.
template <typename F, DataType DT>
concept FillTensorBufferFunctor = requires(const F& f, size_t index) {
    { f(index) } -> std::convertible_to<detail::cpp_type_t<DT>>;
};

/// @brief Utility for initializing tensor buffers.
///
/// This function is a utility for initializing memory backing a tensor buffer. In
/// contrast to `fill_tensor`, this function first extracts the backing space of
/// the tensor, and then invokes the callback for each (flat) index. This function
/// is particular useful for initializing out-of-bounds indices with a known with a
/// known value.
///
/// @tparam DT The tensor element datatype
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param desc The descriptor of the tensor to initialize.
/// @param buffer The memory of the tensor to initialize.
/// @param f A functor used to get the value at a particular index.
///
/// @see FillTensorBufferFunctor
template <DataType DT, size_t RANK>
void fill_tensor_buffer(const TensorDescriptor<DT, RANK>& desc,
                        void* buffer,
                        FillTensorBufferFunctor<DT> auto f)
{
    fill_tensor(desc.get_space_descriptor(), buffer, [f](auto index) { return f(index[0]); });
}

/// @brief Utility for clearing tensor buffers to a particular value.
///
/// This function initializes all memory backing a particular tensor buffer to
/// one specific value, zero by default. Note that this function ignores strides,
/// and clears the entire buffer backing the tensor.
///
/// @tparam DT The tensor element datatype
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param desc The descriptor of the tensor to initialize.
/// @param buffer The memory of the tensor to initialize.
/// @param value The value to initialize the tensor buffer with.
template <DataType DT, size_t RANK>
void clear_tensor_buffer(const TensorDescriptor<DT, RANK>& desc,
                         void* buffer,
                         detail::cpp_type_t<DT> value = detail::cpp_type_t<DT>{0})
{
    fill_tensor_buffer(desc, buffer, [value]([[maybe_unused]] size_t i) { return value; });
}

/// @brief Utility for copying a tensor from one layout to another
///
/// This function copies tensor data from `src_buffer` to `dst_buffer`,
/// changing the layout from `src_desc` to `dst_desc`. Note that the src and
/// dst tensor lengths must be compatible, otherwise this function may write
/// out of bounds.
///
/// @tparam DT The element datatype of both tensors.
/// @tparam RANK The rank (number of spatial dimensions) of the tensors.
///
/// @param src_desc The descriptor of the source tensor to copy from.
/// @param src_buffer The memory of the source tensor.
/// @param dst_desc The descriptor of the destination tensor to copy to.
/// @param dst_buffer The memory of the destination tensor.
template <DataType DT, size_t RANK>
void copy_tensor(const TensorDescriptor<DT, RANK>& src_desc,
                 const void* src_buffer,
                 const TensorDescriptor<DT, RANK>& dst_desc,
                 void* dst_buffer)
{
    assert(src_desc.get_lengths() == dst_desc.get_lengths());
    const auto src_strides = src_desc.get_strides();
    const auto dst_strides = dst_desc.get_strides();
    tensor_foreach(dst_desc.get_lengths(),
                   [src_buffer, dst_buffer, src_strides, dst_strides](const auto& index) {
                       using T            = detail::cpp_type_t<DT>;
                       const auto* src    = static_cast<const T*>(src_buffer);
                       auto* dst          = static_cast<T*>(dst_buffer);
                       const auto src_off = calculate_offset(index, src_strides);
                       const auto dst_off = calculate_offset(index, dst_strides);
                       dst[dst_off]       = src[src_off];
                   });
}

/// @brief Simple iterator implementation over tensors.
///
/// This type implements a simple "iterator" type for tensor types,
/// basically exposing operator[] for flat indices. This type is useful
/// to be able to provide a "pointer-like" object to API that does not
/// expect higher dimensional tensor types, and expects linear pointers
/// instead. Ideally, one just needs to replace the `T* ptr` with
/// `Iterator it` to update those API to be compatible with this type.
///
/// @note This is not intended to be a full implementation of the C++
/// iterator concept. For example, it does not really hold any state,
/// because that is not really useful anyway.
///
/// @tparam DT The datatype of the tensor to iterate over. Note that this
/// is only here for reference purposes, the actual data type of the backing
/// memory is provided via the backing iterator type.
/// @tparam RANK The rank (number of spatial dimensions) of the tensors.
/// @tparam Iterator The backing iterator type. This can be a (non-void)
/// pointer type.
template <DataType DT, size_t RANK, typename Iterator>
struct FlatTensorIterator
{
    /// @brief Construct a FlatTensorIterator.
    ///
    /// Construct a FlatTensorIterator from a tensor descriptor and a backing
    /// iterator. The backing iterator can just be a non-void pointer type,
    /// note that the result of FlatTensorIterator::operator[] is the same as
    /// that of Iterator::operator[].
    ///
    /// @param desc The descriptor of the tensor to iterate.
    /// @param inner The inner iterator, for example a (non-void) pointer.
    FlatTensorIterator(const TensorDescriptor<DT, RANK>& desc, Iterator inner)
        : iter_(desc.get_lengths()), strides_(desc.get_strides()), inner_(inner)
    {
    }

    /// @brief Return the value at a particular flat index.
    ///
    /// This function returns the value of the tensor at flat coordinate
    /// `flat_index`. This index is then unflattened into a multi-dimensional
    /// index according to the way described in `NdIter`, and a tensor offset
    /// is computed from that according to `calculate_offset`. The value at
    /// that offset in the inner iterator is then the return value of this
    /// function.
    ///
    /// @note NdIter iterates such that the inner dimension (right-most value
    /// in the tensor shape) changes fastest.
    ///
    /// @note This function performs no bounds checking.
    ///
    /// @param flat_index The flat index into this tensor.
    ///
    /// @pre flat_index < numel()
    ///
    /// @see NdIter
    __host__ __device__ auto& operator[](size_t flat_index) const
    {
        const auto index  = iter_(flat_index);
        const auto offset = calculate_offset(index, strides_);
        return inner_[offset];
    }

    /// @brief Return the total number of elements to iterate over.
    ///
    /// @see NdIter::numel()
    __host__ __device__ size_t numel() const { return iter_.numel(); }

    private:
    NdIter<RANK> iter_;
    Extent<RANK> strides_;
    Iterator inner_;
};

template <DataType DT, size_t RANK, typename Iterator>
FlatTensorIterator(const TensorDescriptor<DT, RANK>&,
                   Iterator) -> FlatTensorIterator<DT, RANK, Iterator>;

} // namespace ck_tile::builder::test
