#pragma once
#include <hip/hip_runtime.h>
#include <memory>
#include <iostream>
#include <stdexcept>

namespace ck_tile::runtime {

inline void CheckHipError(hipError_t err)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err));
    }
}

template <typename T>
auto AllocDevMem(const size_t n)
{
    auto hip_deleter = [](int* ptr) {
        if(!ptr)
        {
            return;
        }
        if(hipError_t err = hipFree(ptr); err != hipSuccess)
        {
            throw std::runtime_error(std::string("Error during hipFree: ") +
                                     hipGetErrorString(err));
        }
        std::cout << "hipFree called for device memory at " << ptr << std::endl;
    };
    std::unique_ptr<int, decltype(hip_deleter)> d_data(nullptr, hip_deleter);

    // Allocate memory on the device
    void* ptr = nullptr;
    if(hipError_t err = hipMalloc(&ptr, n * sizeof(T)); err != hipSuccess)
    {
        throw std::runtime_error(std::string("Error during hipMalloc: ") + hipGetErrorString(err));
    }
    std::cout << "Allocated device memory at " << ptr << std::endl;

    // Transfer ownership to the unique_ptr
    d_data.reset(static_cast<int*>(ptr));
    return d_data;
}

} // namespace ck_tile::runtime
