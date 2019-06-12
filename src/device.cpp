#include "composable_kernel/utility/config.hpp"
#include "device.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
#if DEVICE_BACKEND_HIP
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
#elif DEVICE_BACKEND_CUDA
    checkCudaErrors(cudaMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
#endif
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

void DeviceMem::ToDevice(const void* p)
{
#if DEVICE_BACKEND_HIP
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
#elif DEVICE_BACKEND_CUDA
    checkCudaErrors(
        cudaMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, cudaMemcpyHostToDevice));
#endif
}

void DeviceMem::FromDevice(void* p)
{
#if DEVICE_BACKEND_HIP
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
#elif DEVICE_BACKEND_CUDA
    checkCudaErrors(cudaMemcpy(p, mpDeviceBuf, mMemSize, cudaMemcpyDeviceToHost));
#endif
}

DeviceMem::~DeviceMem()
{
#if DEVICE_BACKEND_HIP
    hipGetErrorString(hipFree(mpDeviceBuf));
#elif DEVICE_BACKEND_CUDA
    checkCudaErrors(cudaFree(mpDeviceBuf));
#endif
}

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
#if DEVICE_BACKEND_HIP
        hipEventCreate(&mStart);
        hipEventCreate(&mEnd);
#elif DEVICE_BACKEND_CUDA
        cudaEventCreate(&mStart);
        cudaEventCreate(&mEnd);
#endif
    }

    ~KernelTimerImpl()
    {
#if DEVICE_BACKEND_HIP
        hipEventDestroy(mStart);
        hipEventDestroy(mEnd);
#elif DEVICE_BACKEND_CUDA
        cudaEventDestroy(mStart);
        cudaEventDestroy(mEnd);
#endif
    }

    void Start()
    {
#if DEVICE_BACKEND_HIP
        hipEventRecord(mStart, 0);
#elif DEVICE_BACKEND_CUDA
        cudaEventRecord(mStart, 0);
#endif
    }

    void End()
    {
#if DEVICE_BACKEND_HIP
        hipEventRecord(mEnd, 0);
        hipEventSynchronize(mEnd);
#elif DEVICE_BACKEND_CUDA
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
#endif
    }

    float GetElapsedTime() const
    {
        float time;
#if DEVICE_BACKEND_HIP
        hipEventElapsedTime(&time, mStart, mEnd);
#elif DEVICE_BACKEND_CUDA
        cudaEventElapsedTime(&time, mStart, mEnd);
#endif
        return time;
    }

#if DEVICE_BACKEND_HIP
    hipEvent_t mStart, mEnd;
#elif DEVICE_BACKEND_CUDA
    cudaEvent_t mStart, mEnd;
#endif
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
