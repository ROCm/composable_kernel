#pragma once
#include <algorithm>
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

    __host__ __device__ unsigned long GetStride(unsigned i) const { return mpStrides[i]; }

    // this is ugly
    __host__ __device__ unsigned long
    Get1dIndex(unsigned n, unsigned c, unsigned h, unsigned w) const
    {
        return n * mpStrides[0] + c * mpStrides[1] + h * mpStrides[2] + w * mpStrides[3];
    }

    unsigned mpLengths[NDim];
    unsigned long mpStrides[NDim];
};
