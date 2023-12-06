#pragma once

#include "ck/ck.hpp"

// #include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/dynamic_buffer.hpp"
#include "ck/utility/amd_address_space.hpp"
// #include "ck/utility/sequence.hpp"
// #include "ck/utility/sequence_helper.hpp"
// #include "ck/utility/is_detected.hpp"

// #include "ck/tensor_description/tensor_descriptor.hpp"
// #include "ck/tensor_description/tensor_descriptor_helper.hpp"
// #include "ck/tensor_description/multi_index_transform_helper.hpp"

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
          typename Strides>
struct Tensor;
/// @endcond

template <typename PointerElementType>
struct MemoryPointerTag
{
    MemoryPointerTag(PointerElementType* pointer) : pointer_(pointer) {}

    using ElementType = PointerElementType;

    AddressSpaceEnum buffer_adress_space_ = AddressSpaceEnum::Generic;
    ElementType* pointer_;
};

template <typename ElementType>
struct GlobalMemoryPointerTag : public MemoryPointerTag<ElementType>
{
    AddressSpaceEnum buffer_adress_space_ = AddressSpaceEnum::Global;
};

template <typename ElementType>
struct SharedMemoryPointerTag : public MemoryPointerTag<ElementType>
{
    AddressSpaceEnum buffer_adress_space_ = AddressSpaceEnum::Lds;
};

template <typename ElementType>
struct SgprMemoryPointerTag : public MemoryPointerTag<ElementType>
{
    AddressSpaceEnum buffer_adress_space_ = AddressSpaceEnum::Sgpr;
};

template <typename ElementType>
struct VgprMemoryPointerTag : public MemoryPointerTag<ElementType>
{
    AddressSpaceEnum buffer_adress_space_ = AddressSpaceEnum::Vgpr;
};

template <typename ElementType>
constexpr auto make_gmem_ptr(const ElementType* pointer)
{
    return GlobalMemoryPointerTag<ElementType>(pointer);
}

template <typename ElementType>
constexpr auto make_smem_ptr(const ElementType* pointer)
{
    return SharedMemoryPointerTag<ElementType>(pointer);
}

template <typename ElementType>
constexpr auto make_sgprmem_ptr(const ElementType* pointer)
{
    return SgprMemoryPointerTag<ElementType>(pointer);
}

template <typename ElementType>
constexpr auto make_vgprmem_ptr(const ElementType* pointer)
{
    return VgprMemoryPointerTag<ElementType>(pointer);
}

template <typename ElementType, typename Shape, typename Strides>
constexpr auto make_tensor(ElementType* pointer, const Layout<Shape, Strides>& layout)
{
    return Tensor<AddressSpaceEnum::Generic, ElementType, Shape, Strides>(pointer, layout);
}

template <typename ElementType, typename Shape, typename Strides>
constexpr auto make_tensor(MemoryPointerTag<ElementType>& mem_tag,
                           const Layout<Shape, Strides>& layout)
{
    return Tensor<mem_tag.buffer_adress_space_, ElementType, Shape, Strides>(mem_tag.pointer_,
                                                                             layout);
}

} // namespace wrapper
} // namespace ck
