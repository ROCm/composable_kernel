#pragma once
#include "helper_cuda.h"
#include "tensor.hpp"

struct DeviceTensorDescriptor
{
    DeviceTensorDescriptor() = delete;

    __host__ DeviceTensorDescriptor(const TensorDescriptor& host_desc)
        : mDataType(host_desc.GetDataType()), mDim(host_desc.GetDimension())
    {
        std::size_t data_sz = host_desc.GetDataType() == DataType_t::Float ? 4 : 2;

        checkCudaErrors(cudaMalloc(&mpLengths, data_sz * mDim));
        checkCudaErrors(cudaMalloc(&mpStrides, data_sz * mDim));

        checkCudaErrors(
            cudaMemcpy(const_cast<void*>(static_cast<const void*>(host_desc.GetLengths().data())),
                       mpLengths,
                       data_sz * mDim,
                       cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(const_cast<void*>(static_cast<const void*>(host_desc.GetStrides().data())),
                       mpStrides,
                       data_sz * mDim,
                       cudaMemcpyHostToDevice));
    }

    __host__ ~DeviceTensorDescriptor()
    {
        checkCudaErrors(cudaFree(mpLengths));
        checkCudaErrors(cudaFree(mpStrides));
    }

    DataType_t mDataType;
    unsigned long mDim;
    unsigned long* mpLengths;
    unsigned long* mpStrides;
};
