#ifndef CK_STATIC_TENSOR_HPP
#define CK_STATIC_TENSOR_HPP

#include "ignore.hpp"

namespace ck {

// StaticTensor for Scalar
template <AddressSpaceEnum_t AddressSpace,
          typename T,
          typename TensorDesc,
          bool InvalidElementUseNumericalZeroValue,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
struct StaticTensor
{
    static constexpr auto desc_                  = TensorDesc{};
    static constexpr index_t ndim_               = TensorDesc::GetNumOfDimension();
    static constexpr index_t element_space_size_ = desc_.GetElementSpaceSize();

    __host__ __device__ constexpr StaticTensor() : invalid_element_value_{0} {}

    __host__ __device__ constexpr StaticTensor(T invalid_element_value)
        : invalid_element_value_{invalid_element_value}
    {
    }

    // read access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr const T& operator[](Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_[Number<offset>{}];
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

    // write access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr T& operator()(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_(Number<offset>{});
        }
        else
        {
            return ignore;
        }
    }

    StaticBuffer<AddressSpace, T, element_space_size_, true> data_;
    T invalid_element_value_ = T{0};
};

// StaticTensor for vector
template <AddressSpaceEnum_t AddressSpace,
          typename T,
          index_t ScalarPerVector,
          typename TensorDesc,
          bool InvalidElementUseNumericalZeroValue,
          typename enable_if<TensorDesc::IsKnownAtCompileTime(), bool>::type = false>
struct StaticTensorTupleOfVectorBuffer
{
    static constexpr auto desc_                  = TensorDesc{};
    static constexpr index_t ndim_               = TensorDesc::GetNumOfDimension();
    static constexpr index_t element_space_size_ = desc_.GetElementSpaceSize();

    static constexpr index_t num_of_vector_ =
        math::integer_divide_ceil(element_space_size_, ScalarPerVector);

    __host__ __device__ constexpr StaticTensorTupleOfVectorBuffer() : invalid_element_value_{0} {}

    __host__ __device__ constexpr StaticTensorTupleOfVectorBuffer(T invalid_element_value)
        : invalid_element_value_{invalid_element_value}
    {
    }

    // read access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr const T& operator[](Idx) const
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_[Number<offset>{}];
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

    // write access
    template <typename Idx,
              typename enable_if<is_known_at_compile_time<Idx>::value && Idx::Size() == ndim_,
                                 bool>::type = false>
    __host__ __device__ constexpr T& operator()(Idx)
    {
        constexpr auto coord = make_tensor_coordinate(desc_, to_multi_index(Idx{}));

        constexpr index_t offset = coord.GetOffset();

        constexpr bool is_valid = coordinate_has_valid_offset(desc_, coord);

        if constexpr(is_valid)
        {
            return data_(Number<offset>{});
        }
        else
        {
            return ignore;
        }
    }

    StaticBufferTupleOfVector<AddressSpace, T, num_of_vector_, ScalarPerVector, true> data_;
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
