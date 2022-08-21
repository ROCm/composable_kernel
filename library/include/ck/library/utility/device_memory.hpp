// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>

#include <hip/hip_runtime.h>

#include "ck/utility/data_type.hpp"

template <typename T>
__global__ void set_buffer_value(T* p, T x, uint64_t buffer_element_size)
{
    for(uint64_t i = threadIdx.x; i < buffer_element_size; i += blockDim.x)
    {
        p[i] = x;
    }
}

class DeviceMem
{
    void ToDeviceImpl(const void* p) const;
    void FromDeviceImpl(void* p) const;

    public:
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer() const;
    std::size_t GetBufferSize() const;
    void SetZero() const;
    template <typename T>
    void SetValue(T x) const;
    ~DeviceMem();

    template <typename T>
    std::enable_if_t<!std::is_const_v<T> &&
                     (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, ck::bhalf_t> || std::is_same_v<T, ck::half_t> ||
                      std::is_same_v<T, float> || std::is_same_v<T, double>)>
    FromDevice(T* host) const
    {
        assert(device.GetBufferSize() % sizeof(T) == 0);

        FromDeviceImpl(host);
    }

    template <typename T>
    std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t> ||
                     std::is_same_v<T, ck::bhalf_t> || std::is_same_v<T, ck::half_t> ||
                     std::is_same_v<T, ck::half_t> || std::is_same_v<T, float> ||
                     std::is_same_v<T, double>>
    ToDevice(const T* host) const
    {
        assert(device.GetBufferSize() % sizeof(T) == 0);

        ToDeviceImpl(host);
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    void ToDevice(const ck::int4_t* host) const;
    void FromDevice(ck::int4_t* host) const;
#endif

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

template <typename T>
void DeviceMem::SetValue(T x) const
{
    if(mMemSize % sizeof(T) != 0)
    {
        throw std::runtime_error("wrong! not entire DeviceMem will be set");
    }

    set_buffer_value<T><<<1, 1024>>>(static_cast<T*>(mpDeviceBuf), x, mMemSize / sizeof(T));
}
