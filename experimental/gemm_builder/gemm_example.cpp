#include <iostream>

#include <hip/hip_runtime.h>

#include "gemm_builder.h"
#include "utils.hpp"

namespace ckb = ck_tile::builder;

namespace example {

// Reproduce example/ck_tile/03_gemm/universal_gemm.cpp
//
// That example is kind of hard to follow, but the basic idea is that
// the function invoke_gemm (in run_gemm_example.inc) gets called with
// a GemmConfigComputeV3 (in gemm_utils.hpp), which calls the function
// gemm (in universal_gemm.cpp).

struct MyGemmTypes
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using CDataType   = ck_tile::bf16_t;
    using AccDataType = float;
};

struct MyGemmLayout
{
    using ALayout = ckb::RowMajor;
    using BLayout = ckb::ColMajor;
    using CLayout = ckb::RowMajor;
};

using Builder = ckb::GemmBuilder<MyGemmTypes, MyGemmLayout>;
using Kernel  = Builder::Kernel;

} // namespace example

int main()
{
    // Describe the GEMM kernel.
    std::cout << "Kernel name: " << example::Kernel::GetName() << std::endl;
    std::cout << "Shape:       " << example::Builder::GemmShape::GetName() << std::endl;
    std::cout << "Problem:     " << example::Builder::UniversalGemmProblem::GetName() << std::endl;
    std::cout << "Pipeline:    " << example::Builder::GemmPipeline::GetName() << std::endl;

    // Execute the GEMM kernel.
    try
    {
        const int M = 16, N = 64, K = 128;

        auto a_dev = example::AllocDevMem<ck_tile::bf16_t>(M * K);
        auto b_dev = example::AllocDevMem<ck_tile::bf16_t>(K * N);
        auto c_dev = example::AllocDevMem<ck_tile::bf16_t>(M * N);

        auto kernel_args = ck_tile::UniversalGemmKernelArgs{
            .as_ptr    = {a_dev.get()}, // As input tensor's device pointer(s)
            .bs_ptr    = {b_dev.get()}, // Bs input tensor's device pointer(s)
            .ds_ptr    = {},            // Ds input tensor's device pointer(s) (empty if unused)
            .e_ptr     = c_dev.get(),   // E output tensor's device pointer
            .M         = M,             // GEMM's M dimension size
            .N         = N,             // GEMM's N dimension size
            .K         = K,             // GEMM's K dimension size
            .stride_As = {M},           // Stride(s) for As tensor(s)
            .stride_Bs = {N},           // Stride(s) for Bs tensor(s)
            .stride_Ds = {},            // Stride(s) for Ds tensor(s) (empty if unused)
            .stride_E  = M,             // Stride for E tensor
            .k_batch   = 1              // Batch size (for batched GEMM)
        };
        if(!example::Kernel::IsSupportedArgument(kernel_args))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        dim3 grid_dim  = example::Kernel::GridSize(M, N, 1);
        dim3 block_dim = example::Kernel::BlockSize();

        std::cout << "Running " << M << " x " << N << " x " << K << " GEMM kernel..." << std::endl;
        std::cout << "Grid size:  " << grid_dim.x << " x " << grid_dim.y << " x " << grid_dim.z
                  << std::endl;
        std::cout << "Block size: " << block_dim.x << " x " << block_dim.y << " x " << block_dim.z
                  << std::endl;

        ckb::launch_kernel<example::Kernel>
            <<<grid_dim, block_dim, 0, hipStreamDefault>>>(kernel_args);

        example::CheckHipError(hipDeviceSynchronize());
        std::cout << "GEMM completed successfully!" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
