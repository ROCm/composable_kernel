#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <memory>
#include "config.hpp"

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
float launch_kernel(F kernel, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    KernelTimer timer;

#if CK_DEVICE_BACKEND_AMD
    timer.Start();

    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, 0, args...);

    timer.End();

    hipGetErrorString(hipGetLastError());
#elif CK_DEVICE_BACKEND_NVIDIA
    const void* f  = reinterpret_cast<const void*>(kernel);
    void* p_args[] = {&args...};

    timer.Start();

    cudaError_t error = cudaLaunchKernel(f, grid_dim, block_dim, p_args, lds_byte, 0);

    timer.End();

    checkCudaErrors(error);
#endif

    return timer.GetElapsedTime();
}

#endif
