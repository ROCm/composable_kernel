// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/dynamic_buffer.hpp"
#include "ck/utility/amd_address_space.hpp"

namespace ck {
namespace wrapper {

// Disable from doxygen docs generation
/// @cond
// forward declarations
template <typename Shape, typename Strides>
struct Layout;
template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,     // params for Register memory
          index_t ScalarPerVector // param for Register memory
          >

struct Tensor;

template <typename FromType, typename ToType>
struct Slice
{
    __host__ __device__ constexpr Slice() : from_(), to_() {}
    __host__ __device__ constexpr Slice(FromType from, ToType to) : from_(from), to_(to) {}

    template <typename T>
    __host__ __device__ constexpr auto range(const T& dim) const
    {
        if constexpr(is_same_v<FromType, index_t> || is_same_v<ToType, index_t> ||
                     is_same_v<T, index_t>)
        {
            assert(dim >= to_ && from_ >= 0 && (to_ < 0 || to_ > from_) && "Invalid range");
            if(to_ < 0)
            {
                return dim - from_ + to_ + 1;
            }
            else
            {
                // workaround if one of end of interval is index_t and secound one Number<>
                return static_cast<index_t>(to_) - static_cast<index_t>(from_);
            }
        }
        else
        {
            static_assert(dim >= to_ && from_ >= Number<0>{} && (to_ < 0 || to_ > from_),
                          "Invalid range");
            if constexpr(to_ < 0)
            {
                return dim - from_ + to_ + Number<1>{};
            }
            else
            {
                return to_ - from_;
            }
        }
    }

    __host__ __device__ static constexpr bool IsSlice() { return true; }

    const FromType from_;
    const ToType to_;
};

template <typename T>
using is_slice = decltype(std::declval<T&>().IsSlice());

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());
/// @endcond

/**
 * \brief Memory type, allowed members:
 * - Generic,
 * - Global,
 * - Lds,
 * - Sgpr,
 * - Vgpr,
 */
using MemoryTypeEnum = AddressSpaceEnum;

/**
 * \brief Make tensor function.
 *
 * \tparam MemoryType Type of memory.
 * \param pointer Pointer to the memory.
 * \param layout Tensor layout.
 * \return Constructed tensor.
 */
template <MemoryTypeEnum MemoryType, typename ElementType, typename Shape, typename Strides>
constexpr auto make_tensor(ElementType* pointer, const Layout<Shape, Strides>& layout)
{
    return Tensor<MemoryType, ElementType, Shape, Strides, 0 /*NumVectors*/, 0 /*ScalarPerVector*/>(
        pointer, layout);
}

/**
 * \brief Make sgpr or vpgr tensor function.
 *
 * \tparam MemoryType Type of memory.
 * \tparam NumVectors Number of vectors.
 * \tparam ScalarPerVector Scalars per vector.
 * \tparam ElementType Memory data type.
 * \param layout Tensor layout.
 * \return Constructed tensor.
 */
template <MemoryTypeEnum MemoryType,
          index_t NumVectors,
          index_t ScalarPerVector,
          typename ElementType,
          typename Shape,
          typename Strides>
constexpr auto make_register_tensor(const Layout<Shape, Strides>& layout)
{
    static_assert(!IsNestedTuple(Shape{}), "Register tensor with nested layout is not supported");
    return Tensor<MemoryType, ElementType, Shape, Strides, NumVectors, ScalarPerVector>(layout);
}

/**
 * \brief Get Tensor Layout.
 *
 * \param tensor Tensor to get layout.
 * \return Requsted layout.
 */
template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr auto&
layout(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
           tensor)
{
    return tensor.GetLayout();
}

/**
 * \brief Product of tensor shape dims.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get Shape.
 * \return Requsted size.
 */
template <index_t... Idxs,
          AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr index_t
size(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
         tensor)
{
    return size<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Rank of Shape tuple.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get rank.
 * \return Requsted rank.
 */
template <index_t... Idxs,
          AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr index_t
rank(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
         tensor)
{
    return rank<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Depth of Shape tuple.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get depth.
 * \return Requsted depth.
 */
template <index_t... Idxs,
          AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr index_t
depth(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
          tensor)
{
    return depth<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Get Tensor strides.
 *
 * \param tensor Tensor to get strides.
 * \return Requsted strides.
 */
template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr auto&
stride(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
           tensor)
{
    return stride(tensor.GetLayout());
}

/**
 * \brief Get Tensor shape.
 *
 * \param tensor Tensor to get shape.
 * \return Requsted shape.
 */
template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,
          index_t ScalarPerVector>
__host__ __device__ constexpr auto&
shape(const Tensor<BufferAddressSpace, ElementType, Shape, Strides, NumVectors, ScalarPerVector>&
          tensor)
{
    return shape(tensor.GetLayout());
}

/**
 * \brief Get dim slice.
 *
 * \param from Beginning of the interval.
 * \param to End of the interval. (could be also negative to index from the end)
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
template <typename FromType, typename ToType>
constexpr auto slice(const FromType from, const ToType to)
{
    return Slice<FromType, ToType>(from, to);
}

/**
 * \brief Get dim slice. (Assumed that from is equal to 1)
 *
 * \param to End of the interval. (could be also negative to index from the end)
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
template <typename ToType>
constexpr auto slice(const ToType to)
{
    if constexpr(is_same_v<ToType, index_t>)
    {
        return Slice<index_t, ToType>(0, to);
    }
    else
    {
        return Slice<Number<0>, ToType>(Number<0>{}, to);
    }
}

/**
 * \brief Get whole dim slice (from = 0, to = -1).
 *
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
constexpr auto slice() { return Slice<Number<0>, Number<-1>>(Number<0>{}, Number<-1>{}); }

} // namespace wrapper
} // namespace ck
