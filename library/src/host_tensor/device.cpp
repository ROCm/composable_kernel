#include <chrono>
#include "device.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() { return mMemSize; }

void DeviceMem::ToDevice(const void* p)
{
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

void DeviceMem::SetZero() { hipGetErrorString(hipMemset(mpDeviceBuf, 0, mMemSize)); }

DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }

DeviceAlignedMemCPU::DeviceAlignedMemCPU(std::size_t mem_size, std::size_t alignment)
    : mMemSize(mem_size), mAlignment(alignment)
{
    assert(!(alignment == 0 || (alignment & (alignment - 1)))); // check pow of 2

    void* p1;
    void** p2;
    int offset = alignment - 1 + sizeof(void*);
    p1         = malloc(mem_size + offset);
    assert(p1 != nullptr);

    p2     = reinterpret_cast<void**>((reinterpret_cast<size_t>(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    mpDeviceBuf = reinterpret_cast<void*>(p2);
}

void* DeviceAlignedMemCPU::GetDeviceBuffer() { return mpDeviceBuf; }

std::size_t DeviceAlignedMemCPU::GetBufferSize() { return mMemSize; }

void DeviceAlignedMemCPU::SetZero() { memset(mpDeviceBuf, 0, mMemSize); }

DeviceAlignedMemCPU::~DeviceAlignedMemCPU() { free((reinterpret_cast<void**>(mpDeviceBuf))[-1]); }

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
        hipGetErrorString(hipEventCreate(&mStart));
        hipGetErrorString(hipEventCreate(&mEnd));
    }

    ~KernelTimerImpl()
    {
        hipGetErrorString(hipEventDestroy(mStart));
        hipGetErrorString(hipEventDestroy(mEnd));
    }

    void Start()
    {
        hipGetErrorString(hipDeviceSynchronize());
        hipGetErrorString(hipEventRecord(mStart, nullptr));
    }

    void End()
    {
        hipGetErrorString(hipEventRecord(mEnd, nullptr));
        hipGetErrorString(hipEventSynchronize(mEnd));
    }

    float GetElapsedTime() const
    {
        float time;
        hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
        return time;
    }

    hipEvent_t mStart, mEnd;
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }

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
