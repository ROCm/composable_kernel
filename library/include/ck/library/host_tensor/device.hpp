#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "ck/options.hpp"

#include <memory>
#include <functional>
#include <thread>
#include <chrono>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"


inline void hip_check(hipError_t x)
{
    if(x != hipSuccess)
     throw std::runtime_error("Failed to run HIP call");

}

template<typename F, F f>
struct managed_deleter
{
    template<typename T>
    void operator()(T * t)
    {
        if(t != nullptr)
        {
            std::ignore = f(t);
        }
    }

};

template<typename T, typename F, F f>
using managed_pointer = std::unique_ptr<T, managed_deleter<F, f>>;
using hipEventPtr = managed_pointer<typename std::remove_pointer<hipEvent_t>::type, decltype(&hipEventDestroy), hipEventDestroy>;

inline hipEventPtr make_hip_event()
{
    hipEvent_t result = nullptr;
    hip_check(hipEventCreate(&result));
    return hipEventPtr{result};
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

using device_stream_t = hipStream_t;

template <typename... Args, typename F>
void launch_kernel(F kernel, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, hipStream_t stream_id, Args... args)
{
    hipLaunchKernelGGL(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);
}

template <typename... Args, typename F>
float launch_and_time_kernel(
    F kernel, int nrepeat, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, hipStream_t stream_id, bool measure_time, Args... args)
{
#if CK_TIME_KERNELS
    KernelTimer timer;

    printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
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
#else
    std::ignore = nrepeat;
    hipEventPtr start = nullptr;
    hipEventPtr stop = nullptr;
    float elapsed_time = 0.0f;
    if(measure_time)
    {
        start = make_hip_event();
        stop = make_hip_event();
        hip_check(hipEventRecord(start.get(), stream_id));
    }

    launch_kernel(kernel, grid_dim, block_dim, lds_byte, stream_id, args...);

    if(measure_time)
    {
        hip_check(hipEventRecord(stop.get(), stream_id));
        hip_check(hipEventSynchronize(stop.get()));
        hip_check(hipEventElapsedTime(&elapsed_time, start.get(), stop.get()));
    }

    return elapsed_time;
#endif
}
#endif
