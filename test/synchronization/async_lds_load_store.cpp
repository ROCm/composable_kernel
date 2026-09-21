// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"

using ::ck::DeviceMem;
using F8DataType = ck::f8_t;

// Very specific M and K values for illustrative purposes
constexpr int K = 16; // rows must be 128 bit aligned
constexpr int M = 32;

/**
 * \brief async load 32x16 matrix from \a in into LDS, store LDS data into \a out.
 */
__global__ void async_load_store_kernel(F8DataType* in, F8DataType* out, const int stride)
{
    // basic consistency check
    if(K > stride)
        return;

    __shared__ F8DataType shared_mem[M * K];

    int tid          = threadIdx.x;
    int global_index = tid * stride;
    int lds_index    = tid * K;

    __attribute__((address_space(3))) F8DataType* lds_ptr =
        reinterpret_cast<__attribute__((address_space(3))) F8DataType*>(
            reinterpret_cast<uintptr_t>(shared_mem + lds_index));
    __attribute__((address_space(1))) F8DataType* g_ptr_in =
        reinterpret_cast<__attribute__((address_space(1))) F8DataType*>(
            reinterpret_cast<uintptr_t>(in + global_index));
    __attribute__((address_space(1))) F8DataType* g_ptr_out =
        reinterpret_cast<__attribute__((address_space(1))) F8DataType*>(
            reinterpret_cast<uintptr_t>(out + global_index));

    ck::amd_async_copy_to_lds_impl<F8DataType, K, 0, false>(g_ptr_in, 0, lds_ptr);

    ck::block_sync_lds_async_load();

    ck::amd_async_store_to_global_impl<F8DataType, K>(lds_ptr, g_ptr_out);

    ck::block_sync_lds_async_load();
}

TEST(SYNCHRONIZATION, AsyncLDSLoadStore)
{
    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, 0));

    const uint32_t mat_size = M * K; // M=32, K=16

    DeviceMem in(mat_size * sizeof(F8DataType));
    std::vector<F8DataType> in_host(mat_size);
    // Initialize the input data
    for(uint32_t i = 0; i < M; ++i)
    {
        for(uint32_t j = 0; j < K; ++j)
        {
            in_host[i * K + j] =
                ck::type_convert<F8DataType>(static_cast<float>(i * K + j) / 10.0f);
        }
    }
    in.ToDevice(in_host.data());

    DeviceMem out(mat_size * sizeof(F8DataType));
    out.SetZero();

    const uint32_t THREADS_PER_BLOCK_X = 32;
    const uint32_t THREADS_PER_BLOCK_Y = 1;
    const uint32_t THREADS_PER_BLOCK_Z = 1;
    const uint32_t GRID_X              = 1;
    const uint32_t GRID_Y              = 1;
    dim3 dimGrid(GRID_X, GRID_Y);
    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    // Launching kernel from host
    async_load_store_kernel<<<dimGrid, dimBlock>>>(static_cast<F8DataType*>(in.GetDeviceBuffer()),
                                                   static_cast<F8DataType*>(out.GetDeviceBuffer()),
                                                   K);
    HIP_CHECK_ERROR(hipGetLastError());

    // Memory transfer from device to host
    std::vector<F8DataType> out_host(mat_size);
    out.FromDevice(out_host.data());

    for(ck::index_t i = 0; i < M; ++i)
    {
        for(ck::index_t j = 0; j < K; ++j)
        {
            EXPECT_EQ(in_host[i * K + j], out_host[i * K + j]);
        }
    }
}
