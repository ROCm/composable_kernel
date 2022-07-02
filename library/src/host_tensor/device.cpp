#include <chrono>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include "device.hpp"

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

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
        hip_check_error(hipEventCreate(&mStart));
        hip_check_error(hipEventCreate(&mEnd));
    }

    ~KernelTimerImpl()
    {
        hip_check_error(hipEventDestroy(mStart));
        hip_check_error(hipEventDestroy(mEnd));
    }

    void Start()
    {
        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(mStart, nullptr));
    }

    void End()
    {
        hip_check_error(hipEventRecord(mEnd, nullptr));
        hip_check_error(hipEventSynchronize(mEnd));
    }

    float GetElapsedTime() const
    {
        float time;
        hip_check_error(hipEventElapsedTime(&time, mStart, mEnd));
        return time;
    }

    hipEvent_t mStart, mEnd;
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
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

struct WallTimerImpl
{
    void Start() { mStart = std::chrono::high_resolution_clock::now(); }

    void End() { mStop = std::chrono::high_resolution_clock::now(); }

    float GetElapsedTime() const
    {
        return static_cast<float>(
                   std::chrono::duration_cast<std::chrono::microseconds>(mStop - mStart).count()) *
               1e-3;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
};

WallTimer::WallTimer() : impl(new WallTimerImpl()) {}

WallTimer::~WallTimer() {}

void WallTimer::Start() { impl->Start(); }

void WallTimer::End() { impl->End(); }

float WallTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
