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

        checkCudaErrors(cudaMemcpy(
            mpLengths, host_desc.GetLengths().data(), data_sz * mDim, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            mpStrides, host_desc.GetStrides().data(), data_sz * mDim, cudaMemcpyHostToDevice));
    }

    __host__ ~DeviceTensorDescriptor()
    {
#if 0
        if(mpLengths != nullptr)
            checkCudaErrors(cudaFree(mpLengths));
        if(mpStrides != nullptr)
            checkCudaErrors(cudaFree(mpStrides));
#endif
    }

    DataType_t mDataType;
    unsigned long mDim;
    unsigned long* mpLengths = nullptr;
    unsigned long* mpStrides = nullptr;
};
