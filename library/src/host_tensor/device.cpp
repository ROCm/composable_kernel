#include "device.hpp"

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
