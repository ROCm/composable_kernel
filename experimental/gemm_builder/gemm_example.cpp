#include <iostream>

#include <hip/hip_runtime.h>

#include "gemm_builder.h"
#include "run_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckr = ck_tile::runtime;

namespace example {

// Reproduce example/ck_tile/03_gemm/universal_gemm.cpp

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

} // namespace example

int main()
{
    // Describe the GEMM kernel.
    std::cout << "Kernel name: " << example::Builder::GetKernelName() << std::endl;
    std::cout << "Shape:       " << example::Builder::GemmShape::GetName() << std::endl;
    std::cout << "Problem:     " << example::Builder::UniversalGemmProblem::GetName() << std::endl;
    std::cout << "Pipeline:    " << example::Builder::GemmPipeline::GetName() << std::endl;

    // Execute the GEMM kernel.
    try
    {
        const int M = 16, N = 64, K = 128;

        auto a_dev = ckr::AllocDevMem<ck_tile::bf16_t>(M * K);
        auto b_dev = ckr::AllocDevMem<ck_tile::bf16_t>(K * N);
        auto c_dev = ckr::AllocDevMem<ck_tile::bf16_t>(M * N);

        auto kernel_args = example::Builder::KernelArgs{
            .as_ptr    = {a_dev.get()}, // Address of tensor A in device memory.
            .bs_ptr    = {b_dev.get()}, // Address of tensor B in device memory.
            .ds_ptr    = {},            // Unused.
            .e_ptr     = c_dev.get(),   // Address of tensor C in device memory.
            .M         = M,             // GEMM's M dimension size
            .N         = N,             // GEMM's N dimension size
            .K         = K,             // GEMM's K dimension size
            .stride_As = {M},           // Stride for tensor A_MK (row major).
            .stride_Bs = {N},           // Stride for tensor B_KN (column major).
            .stride_Ds = {},            // Unused.
            .stride_E  = M,             // Stride for tensor C_MN (row major).
            .k_batch   = 1              // Batch size is 1 for a single GEMM.
        };
        if(!example::Builder::Supports(kernel_args))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        dim3 grid_dim  = example::Builder::GridDim(M, N, 1);
        dim3 block_dim = example::Builder::BlockDim();

        std::cout << "Running " << M << " x " << N << " x " << K << " GEMM kernel..." << std::endl;
        std::cout << "Grid size:  " << grid_dim.x << " x " << grid_dim.y << " x " << grid_dim.z
                  << std::endl;
        std::cout << "Block size: " << block_dim.x << " x " << block_dim.y << " x " << block_dim.z
                  << std::endl;

        ckb::Kernel<example::Builder><<<grid_dim, block_dim, 0, hipStreamDefault>>>(kernel_args);

        ckr::CheckHipError(hipDeviceSynchronize());
        std::cout << "GEMM completed successfully!" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
