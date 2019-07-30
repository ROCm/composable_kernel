#ifndef CK_TENSOR_COORDINATE_HPP
#define CK_TENSOR_COORDINATE_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"

namespace ck {

template <class TensorDesc>
struct NormalTensorCoordinate
{
    using type             = NormalTensorCoordinate;
    using tensor_desc_type = TensorDesc;

    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();

    __host__ __device__ constexpr NormalTensorCoordinate(Array<index_t, nDim> tensor_index)
        : mIndex{tensor_index}, mOffset{tensor_desc_type::GetOffsetFromMultiIndex(tensor_index)}
    {
    }

    template <class... Xs>
    __host__ __device__ constexpr NormalTensorCoordinate(Xs... xs)
        : NormalTensorCoordinate(Array<index_t, nDim>{xs...})
    {
    }

    __host__ __device__ constexpr Array<unsigned, nDim> GetIndex() const { return mIndex; }

    __host__ __device__ constexpr index_t GetOffset() const { return mOffset; }

    template <class IDim, bool PositiveDirection>
    __host__ __device__ void
    MoveOnDimension(IDim idim, index_t step_size, integral_constant<bool, PositiveDirection>)
    {
        if(PositiveDirection)
        {
            mIndex(idim) += step_size;
            mOffset += step_size * tensor_desc_type::GetStride(idim);
        }
        else
        {
            mIndex(idim) -= step_size;
            mOffset -= step_size * tensor_desc_type::GetStride(idim);
        }
    }

    // T is Array or Sequence
    template <class T>
    __host__ __device__ type operator+=(T step_sizes)
    {
#if 0
        static_assert(is_same<typename T::data_type, index_t>, "wrong!");
#endif
        static_assert(T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, true>{});
        });

        return *this;
    }

    template <class T>
    __host__ __device__ type operator-=(T step_sizes)
    {
#if 0
        static_assert(is_same<typename T::data_type, index_t>, "wrong!");
#endif
        static_assert(T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, false>{});
        });

        return *this;
    }

    template <class T>
    __host__ __device__ constexpr type operator+(T step_sizes) const
    {
        type coord = *this;
        coord += step_sizes;
        return coord;
    }

    template <class T>
    __host__ __device__ constexpr type operator-(T step_sizes) const
    {
        type coord = *this;
        coord -= step_sizes;
        return coord;
    }

    // reposition point of origin, and return compensated offset
    __host__ __device__ constexpr index_t RepositionOrigin()
    {
        index_t offset_diff = mOffset;

        mIndex  = make_zero_array<index_t, nDim>();
        mOffset = 0;

        return offset_diff;
    }

    // private:
    Array<index_t, nDim> mIndex;
    index_t mOffset;
};

template <class TensorDesc>
struct MergedTensorCoordinate
{
    using type             = MergedTensorCoordinate;
    using tensor_desc_type = TensorDesc;

    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();
    static constexpr index_t nOriginalDim =
        tensor_desc_type::GetOriginalTensorDescriptor().GetNumOfDimension();

    __host__ __device__ constexpr MergedTensorCoordinate(Array<index_t, nDim> tensor_index)
        : mIndex{tensor_index},
          mOriginalIndex{tensor_desc_type::GetOriginalMultiIndexFromMultiIndex(tensor_index)}
    {
        // partial offset on each dimension
        static_for<0, nDim, 1>{}([&](auto idim) {
            constexpr auto partial_original_dims =
                tensor_desc_type::GetContainedOriginalDimensions(idim);

            constexpr auto partial_original_desc =
                tensor_desc_type::GetOriginalTensorDescriptor().Extract(partial_original_dims);

            mPartialOffsets(idim) = partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mOriginalIndex, partial_original_dims));
        });

        // complete offset
        mOffset =
            accumulate_on_array(mPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));
    }

    template <class... Xs>
    __host__ __device__ constexpr MergedTensorCoordinate(Xs... xs)
        : MergedTensorCoordinate(Array<index_t, nDim>{xs...})
    {
    }

    __host__ __device__ constexpr Array<index_t, nDim> GetIndex() const { return mIndex; }

    __host__ __device__ constexpr index_t GetOffset() const { return mOffset; }

    // step_size should be known at compile time
    template <class IDim, bool PositiveDirection>
    __host__ __device__ void
    MoveOnDimension(IDim, index_t step_size, integral_constant<bool, PositiveDirection>)
    {
        constexpr auto idim = IDim{};

        // update multi-index
        if(PositiveDirection)
        {
            mIndex(idim) += step_size;
        }
        else
        {
            mIndex(idim) -= step_size;
        }

        // update rest
        static_if<tensor_desc_type::ContainMultipleOriginalDimensions(idim)>{}([&](auto) {
            constexpr auto partial_original_dims =
                tensor_desc_type::GetContainedOriginalDimensions(idim);

            constexpr index_t ndim_partial_original = partial_original_dims.GetSize();

            constexpr auto partial_original_desc =
                tensor_desc_type::GetOriginalTensorDescriptor().Extract(partial_original_dims);

            const auto partial_original_step_sizes =
                partial_original_desc.GetMultiIndexFrom1dIndex(step_size);

            // update partial original multi-id
            auto partial_original_id = extract_array(mOriginalIndex, partial_original_dims);

            static_if<PositiveDirection>{}([&](auto) {
                partial_original_id += partial_original_step_sizes;

                bool carry = false;

                // do carry check in reversed order, starting from lowest dimension
                // don't check the highest dimension
                static_for<0, ndim_partial_original, 1>{}([&](auto IReverse) {
                    constexpr index_t i = ndim_partial_original - 1 - IReverse;

                    if(carry)
                    {
                        ++partial_original_id(i);
                    }

                    carry = false;

                    if(partial_original_id[i] >= partial_original_desc.GetLength(i))
                    {
                        partial_original_id(i) -= partial_original_desc.GetLength(i);
                        carry = true;
                    }
                });
            }).Else([&](auto) {
                // shift up multi-id to avoid unsigned integer underflow during intermediate
                // calculations. After the shift, should have new_multi_id[...] >= 1
                partial_original_id +=
                    partial_original_desc.GetLengths() - partial_original_step_sizes;

                bool borrow = false;

                // do borrow check in reversed order, starting from lowest dimension
                // don't check the highest dimension
                static_for<0, ndim_partial_original, 1>{}([&](auto IReverse) {
                    constexpr index_t i = ndim_partial_original - 1 - IReverse;

                    if(borrow)
                    {
                        --partial_original_id(i);
                    }

                    borrow = false;

                    if(partial_original_id[i] < partial_original_desc.GetLength(i))
                    {
                        partial_original_id(i) += partial_original_desc.GetLength(i);
                        borrow = true;
                    }
                });

                // shift back down multi-id
                // here, should have new_multi_id[...] >= GetLengths()
                partial_original_id = partial_original_id - partial_original_desc.GetLengths();
            });

            // update "mOriginalIndex"
            static_for<0, ndim_partial_original, 1>{}([&](auto I) {
                constexpr auto idim_original = partial_original_dims[I];

                mOriginalIndex(idim_original) = partial_original_id[I];
            });

            // calculate new partial offset on this merged dimension
            const index_t old_partial_offset = mPartialOffsets[idim];

            mPartialOffsets(idim) =
                partial_original_desc.GetOffsetFromMultiIndex(partial_original_id);

            // update "mThreadSrcOffset", do "+" before "-" to avoid underflow
            mOffset = (mOffset + mPartialOffsets[idim]) - old_partial_offset;
        }).Else([&](auto) {
            constexpr auto idim_original =
                tensor_desc_type::GetContainedOriginalDimensions(idim).Front();

            static_if<PositiveDirection>{}([&](auto fwd) {
                mOriginalIndex(idim_original) += step_size;
                mPartialOffsets(idim) += step_size * fwd(tensor_desc_type{}).GetStride(idim);
                mOffset += step_size * fwd(tensor_desc_type{}).GetStride(idim);
            }).Else([&](auto fwd) {
                mOriginalIndex(idim_original) -= step_size;
                mPartialOffsets(idim) -= step_size * fwd(tensor_desc_type{}).GetStride(idim);
                mOffset -= step_size * fwd(tensor_desc_type{}).GetStride(idim);
            });
        });
    }

    // T is Array or Sequence
    template <class T>
    __host__ __device__ type operator+=(T step_sizes)
    {
#if 0
        static_assert(is_same<typename T::data_type, index_t>, "wrong!");
#endif
        static_assert(T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, true>{});
        });

        return *this;
    }

    template <class T>
    __host__ __device__ type operator-=(T step_sizes)
    {
#if 0
        static_assert(is_same<typename T::data_type, index_t>, "wrong!");
#endif
        static_assert(T::GetSize() == nDim, "wrong!");

        static_for<0, nDim, 1>{}([&](auto idim) {
            this->MoveOnDimension(idim, step_sizes[idim], integral_constant<bool, false>{});
        });

        return *this;
    }

    template <class T>
    __host__ __device__ constexpr type operator+(T step_sizes) const
    {
        type coord = *this;
        coord += step_sizes;
        return coord;
    }

    template <class T>
    __host__ __device__ constexpr type operator-(T step_sizes) const
    {
        type coord = *this;
        coord -= step_sizes;
        return coord;
    }

    // reposition point of origin, and return compensated offset
    __host__ __device__ constexpr index_t RepositionOrigin()
    {
        index_t offset_diff = 0;

        static_for<0, nDim, 1>{}([&](auto idim_) {
            constexpr auto idim = decltype(idim_){};

            static_if<!tensor_desc_type::ContainMultipleOriginalDimensions(idim)>{}([&](auto) {
                constexpr auto idim_original =
                    tensor_desc_type::GetContainedOriginalDimensions(idim).Front();

                mIndex(idim)                  = 0;
                mOriginalIndex(idim_original) = 0;
                mOffset -= mPartialOffsets[idim];
                offset_diff += mPartialOffsets[idim];
                mPartialOffsets(idim) = 0;
            });
        });

        return offset_diff;
    }

    // private:
    Array<index_t, nDim> mIndex;
    Array<index_t, nOriginalDim> mOriginalIndex;
    Array<index_t, nDim> mPartialOffsets; // mPartialOffsets is needed for for unsigned index type
    index_t mOffset;
};

} // namespace ck
#endif
