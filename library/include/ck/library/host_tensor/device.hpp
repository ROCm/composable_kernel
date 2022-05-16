#pragma once

#include <memory>
#include <functional>
#include <thread>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "stream_config.hpp"
#include "ck/options.hpp"

inline void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__ << ": " << __LINE__
           << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}

struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer();
    std::size_t GetBufferSize();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    void SetZero();
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

struct DeviceAlignedMemCPU
{
    DeviceAlignedMemCPU() = delete;
    DeviceAlignedMemCPU(std::size_t mem_size, std::size_t alignment);
    void* GetDeviceBuffer();
    std::size_t GetBufferSize();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    void SetZero();
    ~DeviceAlignedMemCPU();

    void* mpDeviceBuf;
    std::size_t mMemSize;
    std::size_t mAlignment;
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

struct WallTimerImpl;

struct WallTimer
{
    WallTimer();
    ~WallTimer();
    void Start();
    void End();
    float GetElapsedTime() const;

    std::unique_ptr<WallTimerImpl> impl;
};

using device_stream_t = hipStream_t;

template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        const int nrepeat = 10;

        printf("Warm up 1 time\n");

        // warm up
        hipLaunchKernelGGL(
            kernel, grid_dim, block_dim, lds_byte, stream_config.stream_id_, args...);

        printf("Start running %d times...\n", nrepeat);

        KernelTimer timer;
        timer.Start();

        for(int i = 0; i < nrepeat; ++i)
        {
            hipLaunchKernelGGL(
                kernel, grid_dim, block_dim, lds_byte, stream_config.stream_id_, args...);
        }

        timer.End();

        return timer.GetElapsedTime() / nrepeat;
    }
    else
    {
        hipLaunchKernelGGL(
            kernel, grid_dim, block_dim, lds_byte, stream_config.stream_id_, args...);

        return 0;
    }
#else
    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_config.stream_id_, args...);

    return 0;
#endif
}

template <typename... Args, typename F>
void launch_cpu_kernel(F kernel, Args... args)
{
    kernel(args...);
}

template <typename... Args, typename F>
float launch_and_time_cpu_kernel(F kernel, int nrepeat, Args... args)
{
    WallTimer timer;

    int nwarmup = 3;

    for(int i = 0; i < nwarmup; i++)
        kernel(args...);

    timer.Start();
    for(int i = 0; i < nrepeat; i++)
    {
        kernel(args...);
    }
    timer.End();

    return timer.GetElapsedTime() / nrepeat;
}

