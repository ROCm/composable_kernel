#ifndef CK_TENSOR_DESCRIPTOR_HPP
#define CK_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "dimension.hpp"

namespace ck {

template <class Lengths, class Strides>
struct NativeTensorDescriptor
{
    using type                    = NativeTensorDescriptor;
    static constexpr index_t nDim = Lengths::GetSize();

    using Id = MultiIndex<nDim>;

    __host__ __device__ static constexpr auto GetNumOfDimension() { return Number<nDim>{}; }

    __host__ __device__ static constexpr auto GetLengths() { return Lengths{}; }

    __host__ __device__ static constexpr auto GetStrides() { return Strides{}; }

    __host__ __device__ static constexpr auto GetLength(index_t IDim) { return Lengths{}[IDim]; }

    __host__ __device__ static constexpr auto GetStride(index_t IDim) { return Strides{}[IDim]; }

    __host__ __device__ static constexpr index_t GetOffset(Id id)
    {
        // not implemented
    }
};

// LowerTensorDescriptor
// Transforms: std::tuple<DimensionTransforms...>
// LowerIds: std::tuple<Sequence<...>>
// UpperIds: std::tuple<Sequence<...>>
template <class LowTensorDescriptor,
          class Transforms,
          class LowDimensionMasks,
          class UpDimensionMasks>
struct TransformedTensorDescriptor
{
    using type                       = TransformedTensorDescriptor;
    static constexpr index_t nDimUp  = xxxx;
    static constexpr index_t nDimLow = xxx;

    static constexpr index_t nTransform = Transforms::GetSize();

    using UpperId = MultiIndex<nDimUp>;
    using LowerId = MultiIndex<nDimLow>;

    __host__ __device__ static constexpr TransformedTensorDescriptor()
    {
        static_assert(nTransform == Transforms::GetSize() &&
                          nTransform == LowDimensionMasks::GetSize() &&
                          nTransform == UpDimensionMasks::GetSize(),
                      "wrong! # of transformations not the same");

        // TODO: sanity check: LowDimensionMasks should include all low-dimensions,
        //   UpDimensionMasks should include all up-dimensions

        // TODO: sanity check: while a up-dimension could be associated with multille
        // transformation,
        //   a low-dimension should be associated with only one transformation
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetLengths()
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetLowerTensorDescriptor()
    {
        return LowTensorDescriptor{};
    }

    __host__ __device__ static constexpr index_t GetLowerId(UpperId id_up)
    {
        // not implemented
    }

    __host__ __device__ static constexpr index_t GetOffset(UpperId id_up)
    {
        return GetLowerTensorDescriptor().GetOffset(GetLowerId(id_up));
    }

    __host__ __device__ static constexpr auto AreUpperId2OffsetLinear();
    {
        // not implemented
    }

    __host__ __device__ static constexpr auto GetIndependentDimensionGroups()
    {
        // not implemented
    }
};

} // namespace ck
#endif
