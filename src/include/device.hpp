#pragma once
#include <memory>
#include "config.h"

struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

struct KernelTimerImpl;

struct KernelTimer
{
    KernelTimer();
    ~KernelTimer();
    void Start();
    void End();
    float GetElapsedTime() const;

    std::unique_ptr<KernelTimerImpl> impl;
};

template <typename... Args, typename F>
float launch_kernel(F kernel, dim3 grid_dim, dim3 block_dim, Args... args)
{
    KernelTimer timer;

#if DEVICE_BACKEND_HIP
    timer.Start();

    hipLaunchKernelGGL(kernel, grid_dim, block_dim, 0, 0, args...);

    timer.End();

    hipGetErrorString(hipGetLastError());
#elif DEVICE_BACKEND_CUDA
    const void* f = reinterpret_cast<const void*>(kernel);
    void* p_args[]  = {&args...};

    timer.Start();

    cudaError_t error = cudaLaunchKernel(f, grid_dim, block_dim, p_args, 0, 0);

    timer.End();

    checkCudaErrors(error);
#endif

    return timer.GetElapsedTime();
}
