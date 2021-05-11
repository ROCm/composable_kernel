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

#if CK_DEVICE_BACKEND_AMD
using device_stream_t = hipStream_t;

template <typename... Args, typename F>
void launch_kernel(F kernel,
                   dim3 grid_dim,
                   dim3 block_dim,
                   std::size_t lds_byte,
                   hipStream_t stream_id,
                   Args... args)
{
    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);
}

template <typename... Args, typename F>
float launch_and_time_kernel(F kernel,
                             int nrepeat,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             hipStream_t stream_id,
                             Args... args)
{
    KernelTimer timer;

    printf("%s: block_dim {%d, %d, %d}, grid_dim {%d, %d, %d} \n",
           __func__,
           grid_dim.x,
           grid_dim.y,
           grid_dim.z,
           block_dim.x,
           block_dim.y,
           block_dim.z);

    printf("Warm up\n");

    // warm up
    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);

    printf("Start running %d times...\n", nrepeat);

    timer.Start();

    for(int i = 0; i < nrepeat; ++i)
    {
        hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);
    }

    timer.End();

    return timer.GetElapsedTime() / nrepeat;
}

#elif CK_DEVICE_BACKEND_NVIDIA
using device_stream_t = cudaStream_t;

template <typename... Args, typename F>
void launch_kernel(F kernel,
                   dim3 grid_dim,
                   dim3 block_dim,
                   std::size_t lds_byte,
                   cudaStream_t stream_id,
                   Args... args)
{
    const void* f  = reinterpret_cast<const void*>(kernel);
    void* p_args[] = {&args...};

    cudaError_t error = cudaLaunchKernel(f, grid_dim, block_dim, p_args, lds_byte, stream_id);
}

template <typename... Args, typename F>
float launch_and_time_kernel(F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             cudaStream_t stream_id,
                             Args... args)
{
    KernelTimer timer;

    const void* f  = reinterpret_cast<const void*>(kernel);
    void* p_args[] = {&args...};

    timer.Start();

    cudaError_t error = cudaLaunchKernel(f, grid_dim, block_dim, p_args, lds_byte, stream_id);

    timer.End();

    return timer.GetElapsedTime();
}
#endif

#endif
