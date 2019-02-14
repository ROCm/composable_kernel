#include "device.hpp"
#include "cuda_runtime.h"
#include "nvToolsExt.h"
#include "helper_cuda.h"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    checkCudaErrors(cudaMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

void DeviceMem::ToDevice(const void* p)
{
    checkCudaErrors(
        cudaMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, cudaMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    checkCudaErrors(cudaMemcpy(p, mpDeviceBuf, mMemSize, cudaMemcpyDeviceToHost));
}

DeviceMem::~DeviceMem() { checkCudaErrors(cudaFree(mpDeviceBuf)); }

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mEnd);
    }

    ~KernelTimerImpl()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mEnd);
    }

    void Start() { cudaEventRecord(mStart, 0); }

    void End()
    {
        cudaEventRecord(mEnd, 0);
        cudaEventSynchronize(mEnd);
    }

    float GetElapsedTime() const
    {
        float time;
        cudaEventElapsedTime(&time, mStart, mEnd);
        return time;
    }

    cudaEvent_t mStart, mEnd;
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }

void launch_kernel(const void* func, dim3 grid_dim, dim3 block_dim, void** args, float& time)
{
    KernelTimer timer;
    timer.Start();

    cudaError_t error = cudaLaunchKernel(func, grid_dim, block_dim, args, 0, 0);

    timer.End();
    time = timer.GetElapsedTime();

    checkCudaErrors(error);
}
