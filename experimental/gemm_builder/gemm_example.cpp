
// gemm_example.cpp (formerly hello_world.cpp)
#include <iostream>
#include <memory>
#include <hip/hip_runtime.h>

#include "gemm_builder.h"

namespace example {

// Helper to allocate device memory.
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

namespace ckb = ck_tile::builder;

// Reproduce example/ck_tile/03_gemm/universal_gemm.cpp
//
// That example is kind of hard to follow, but the basic idea is that
// the function invoke_gemm (in run_gemm_example.inc) gets called with
// a GemmConfigComputeV3 (in gemm_utils.hpp), which calls the function
// gemm (in universal_gemm.cpp).

struct MyGemmTypes
{
    using ADataType   = float;
    using BDataType   = float;
    using CDataType   = float;
    using AccDataType = float;
};

struct MyGemmLayout
{
    using ALayout = ckb::RowMajor;
    using BLayout = ckb::ColMajor;
    using CLayout = ckb::RowMajor;
};

using Builder = ckb::GemmBuilder<MyGemmTypes, MyGemmLayout>;

using Gemm   = Builder::value;
using Kernel = Builder::Kernel;

} // namespace example

int main()
{
    // Create the GEMM kernel.
    const int M = 1024, N = 2048, K = 64;
    example::Gemm gemm;

    // Describe the GEMM kernel:
    std::cout << "Shape: " << example::Builder::GemmShape::GetName() << std::endl;
    std::cout << "Problem: " << example::Builder::UniversalGemmProblem::GetName() << std::endl;
    // std::cout << "Pipeline: " << example::Builder::GemmPipeline::GetName() << std::endl;
    // std::cout << "Kernel name: " << Kernel::GetName() << std::endl;

    // Try GPU execution.
    try
    {
        auto a_dev = example::AllocDevMem<float>(M * K);
        auto b_dev = example::AllocDevMem<float>(K * N);
        auto c_dev = example::AllocDevMem<float>(M * N);

        gemm.run({.m = M, .n = N, .k = K, .a = a_dev.get(), .b = b_dev.get(), .c = c_dev.get()});
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
