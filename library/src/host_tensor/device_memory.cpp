// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/device_utility/hip_check_error.hpp"
#include "ck/library/host_tensor/device_memory.hpp"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifndef CK_NOGPU
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hip_check_error(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() { return mMemSize; }

void DeviceMem::ToDevice(const void* p)
{
    hip_check_error(hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    hip_check_error(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

void DeviceMem::SetZero() { hip_check_error(hipMemset(mpDeviceBuf, 0, mMemSize)); }

DeviceMem::~DeviceMem() { hip_check_error(hipFree(mpDeviceBuf)); }
#endif

DeviceAlignedMemCPU::DeviceAlignedMemCPU(std::size_t mem_size, std::size_t alignment)
    : mMemSize(mem_size), mAlignment(alignment)
{
    if(mem_size == 0)
    {
        mpDeviceBuf = nullptr;
    }
    else
    {
        assert(!(alignment == 0 || (alignment & (alignment - 1)))); // check pow of 2

        // TODO: posix only
        int rtn = posix_memalign(&mpDeviceBuf, alignment, mem_size);

        assert(rtn == 0);
    }
}

void* DeviceAlignedMemCPU::GetDeviceBuffer() { return mpDeviceBuf; }

std::size_t DeviceAlignedMemCPU::GetBufferSize() { return mMemSize; }

void DeviceAlignedMemCPU::ToDevice(const void* p) { memcpy(mpDeviceBuf, p, mMemSize); }

void DeviceAlignedMemCPU::FromDevice(void* p) { memcpy(p, mpDeviceBuf, mMemSize); }

void DeviceAlignedMemCPU::SetZero() { memset(mpDeviceBuf, 0, mMemSize); }

DeviceAlignedMemCPU::~DeviceAlignedMemCPU()
{
    if(mpDeviceBuf != nullptr)
        free(mpDeviceBuf);
}
