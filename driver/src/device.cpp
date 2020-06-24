#include "config.hpp"
#include "device.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
#if CK_DEVICE_BACKEND_AMD
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
#elif CK_DEVICE_BACKEND_NVIDIA
    cudaMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize);
#endif
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

void DeviceMem::ToDevice(const void* p)
{
#if CK_DEVICE_BACKEND_AMD
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
#elif CK_DEVICE_BACKEND_NVIDIA
    cudaMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, cudaMemcpyHostToDevice);
#endif
}

void DeviceMem::FromDevice(void* p)
{
#if CK_DEVICE_BACKEND_AMD
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
#elif CK_DEVICE_BACKEND_NVIDIA
    cudaMemcpy(p, mpDeviceBuf, mMemSize, cudaMemcpyDeviceToHost);
#endif
}

DeviceMem::~DeviceMem()
{
#if CK_DEVICE_BACKEND_AMD
    hipGetErrorString(hipFree(mpDeviceBuf));
#elif CK_DEVICE_BACKEND_NVIDIA
    cudaFree(mpDeviceBuf);
#endif
}

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
#if CK_DEVICE_BACKEND_AMD
        hipEventCreate(&mStart);
        hipEventCreate(&mEnd);
#elif CK_DEVICE_BACKEND_NVIDIA
        cudaEventCreate(&mStart);
        cudaEventCreate(&mEnd);
#endif
    }

    ~KernelTimerImpl()
    {
#if CK_DEVICE_BACKEND_AMD
        hipEventDestroy(mStart);
        hipEventDestroy(mEnd);
#elif CK_DEVICE_BACKEND_NVIDIA
        cudaEventDestroy(mStart);
        cudaEventDestroy(mEnd);
#endif
    }

    void Start()
    {
#if CK_DEVICE_BACKEND_AMD
        hipDeviceSynchronize();
        hipEventRecord(mStart, 0);
#elif CK_DEVICE_BACKEND_NVIDIA
        cudaDeviceSynchronize();
        cudaEventRecord(mStart, 0);
#endif
    }

    void End()
    {
#if CK_DEVICE_BACKEND_AMD
        hipEventRecord(mEnd, 0);
        hipEventSynchronize(mEnd);
#elif CK_DEVICE_BACKEND_NVIDIA
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
#endif
    }

    float GetElapsedTime() const
    {
        float time;
#if CK_DEVICE_BACKEND_AMD
        hipEventElapsedTime(&time, mStart, mEnd);
#elif CK_DEVICE_BACKEND_NVIDIA
        cudaEventElapsedTime(&time, mStart, mEnd);
#endif
        return time;
    }

#if CK_DEVICE_BACKEND_AMD
    hipEvent_t mStart, mEnd;
#elif CK_DEVICE_BACKEND_NVIDIA
    cudaEvent_t mStart, mEnd;
#endif
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
