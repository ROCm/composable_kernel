// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <memory>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/device_memory.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hip_check_error(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() const { return mMemSize; }

void DeviceMem::ToDeviceImpl(const void* p) const
{
    hip_check_error(hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDeviceImpl(void* p) const
{
    hip_check_error(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
void DeviceMem::ToDevice(const ck::int4_t* host) const
{
    const std::size_t count = GetBufferSize();
    const auto buffer       = std::make_unique<int8_t[]>(count);

    std::copy_n(host, count, buffer.get());

    ToDevice(buffer.get());
}

void DeviceMem::FromDevice(ck::int4_t* host) const
{
    const std::size_t count = GetBufferSize();
    const auto buffer       = std::make_unique<int8_t[]>(count);

    FromDevice(buffer.get());

    std::copy_n(buffer.get(), count, host);
}
#endif

void DeviceMem::SetZero() const { hip_check_error(hipMemset(mpDeviceBuf, 0, mMemSize)); }

DeviceMem::~DeviceMem() { hip_check_error(hipFree(mpDeviceBuf)); }
