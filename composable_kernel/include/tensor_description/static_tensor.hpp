#ifndef CK_STATIC_TENSOR_HPP
#define CK_STATIC_TENSOR_HPP

#include "ignore.hpp"
#include "static_buffer.hpp"

namespace ck {

template <AddressSpaceEnum_t AddressSpace,
          typename T,
          typename TensorDesc,
          bool InvalidElementUseNumericalZeroValue,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
struct StaticTensor
{
    static constexpr index_t NDim         = TensorDesc::GetNumOfDimension();
    static constexpr index_t buffer_size_ = TensorDesc{}.GetElementSpaceSize();

    static constexpr auto desc_ = TensorDesc{};

    using Buffer = StaticBuffer<AddressSpace, T, buffer_size_, InvalidElementUseNumericalZeroValue>;

    __host__ __device__ constexpr StaticTensor() : invalid_element_value_{0} {}

    __host__ __device__ constexpr StaticTensor(T invalid_element_value)
        : invalid_element_value_{invalid_element_value}
    {
    }

    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == NDim,
                                 bool>::type = false>
    __host__ __device__ constexpr const T& operator[](Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return buffer_[Number<offset>{}];
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return T{0};
            }
            else
            {
                return invalid_element_value_;
            }
        }
    }

    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == NDim,
                                 bool>::type = false>
    __host__ __device__ T& operator()(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return buffer_(Number<offset>{});
        }
        else
        {
            return ignore;
        }
    }

    Buffer buffer_;
    T invalid_element_value_ = T{0};
};

template <AddressSpaceEnum_t AddressSpace,
          typename T,
          typename TensorDesc,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
__host__ __device__ constexpr auto make_static_tensor(TensorDesc)
{
    return StaticTensor<AddressSpace, T, TensorDesc, true>{};
}

template <
    AddressSpaceEnum_t AddressSpace,
    typename T,
    typename TensorDesc,
    typename X,
    typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type                   = false,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto make_static_tensor(TensorDesc, X invalid_element_value)
{
    return StaticTensor<AddressSpace, T, TensorDesc, true>{invalid_element_value};
}

} // namespace ck
#endif
