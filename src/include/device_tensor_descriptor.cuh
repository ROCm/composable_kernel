#pragma once
#include <algorithm>
#include "constant_tensor_descriptor.cuh"
#include "helper_cuda.h"
#include "tensor.hpp"

template <unsigned NDim>
struct DeviceTensorDescriptor
{
    __host__ __device__ DeviceTensorDescriptor() = default;

    __host__ DeviceTensorDescriptor(const TensorDescriptor& host_desc)
    {
        assert(NDim == host_desc.GetDimension());
        std::copy(host_desc.GetLengths().begin(), host_desc.GetLengths().end(), mpLengths);
        std::copy(host_desc.GetStrides().begin(), host_desc.GetStrides().end(), mpStrides);
    }

    __host__ __device__ unsigned GetLength(unsigned i) const { return mpLengths[i]; }

    __host__ __device__ unsigned GetStride(unsigned i) const { return mpStrides[i]; }

    // this is ugly, only for 4d
    __host__ __device__ unsigned Get1dIndex(unsigned n, unsigned c, unsigned h, unsigned w) const
    {
        return n * mpStrides[0] + c * mpStrides[1] + h * mpStrides[2] + w * mpStrides[3];
    }

    unsigned mpLengths[NDim];
    unsigned mpStrides[NDim];
};

// this is ugly, only for 4d
template <class TConstTensorDesc>
__host__ __device__ auto make_DeviceTensorDescriptor(TConstTensorDesc)
{
    static_assert(TConstTensorDesc::nDim == 4, "nDim is not 4");

    constexpr auto I0         = Index<0>{};
    constexpr auto I1         = Index<1>{};
    constexpr auto I2         = Index<2>{};
    constexpr auto I3         = Index<3>{};
    constexpr auto const_desc = TConstTensorDesc{};

    constexpr auto ndim = const_desc.GetDimension();

    auto desc = DeviceTensorDescriptor<ndim>{};

    desc.mpLengths[0] = const_desc.GetLength(I0);
    desc.mpLengths[1] = const_desc.GetLength(I1);
    desc.mpLengths[2] = const_desc.GetLength(I2);
    desc.mpLengths[3] = const_desc.GetLength(I3);

    desc.mpStrides[0] = const_desc.GetStride(I0);
    desc.mpStrides[1] = const_desc.GetStride(I1);
    desc.mpStrides[2] = const_desc.GetStride(I2);
    desc.mpStrides[3] = const_desc.GetStride(I3);

    return desc;
}
