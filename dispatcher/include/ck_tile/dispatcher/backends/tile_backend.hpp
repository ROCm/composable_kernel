// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/validation/reference_kernels.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// Kernel instance for CK Tile generated kernels
template <typename SelectedKernel>
class TileKernelInstance : public KernelInstance
{
    public:
    TileKernelInstance(const KernelKey& key, const std::string& name) : key_(key), name_(name) {}

    const KernelKey& get_key() const override { return key_; }

    bool supports(const Problem& problem) const override
    {
        // Check dimension divisibility if padding not enabled
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;

        if(pad_m && pad_n && pad_k)
        {
            // Padding enabled - supports any size
            return true;
        }

        // Check divisibility
        constexpr int tile_m = SelectedKernel::TileM;
        constexpr int tile_n = SelectedKernel::TileN;
        constexpr int tile_k = SelectedKernel::TileK;

        if(!pad_m && problem.M % tile_m != 0)
            return false;
        if(!pad_n && problem.N % tile_n != 0)
            return false;
        if(!pad_k && problem.K % tile_k != 0)
            return false;

        // Check shared memory budget if specified
        if(problem.smem_budget > 0)
        {
            int64_t estimated_smem = estimate_smem_usage();
            if(estimated_smem > problem.smem_budget)
                return false;
        }

        return true;
    }

    std::string get_name() const override { return name_; }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        // Convert void* stream to hipStream_t
        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

        // Construct kernel arguments
        using ADataType = typename SelectedKernel::ADataType;
        using BDataType = typename SelectedKernel::BDataType;
        using CDataType = typename SelectedKernel::CDataType;

        // Note: d_ptrs not yet supported in basic CK Tile kernels
        (void)d_ptrs; // Suppress unused parameter warning

        auto kargs = SelectedKernel::MakeKernelArgs(static_cast<const ADataType*>(a_ptr),
                                                    static_cast<const BDataType*>(b_ptr),
                                                    static_cast<CDataType*>(c_ptr),
                                                    problem.M,
                                                    problem.N,
                                                    problem.K,
                                                    problem.k_batch);

        // Validate arguments
        if(!SelectedKernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Kernel does not support the given arguments");
        }

        // Calculate grid and block dimensions
        dim3 grids       = SelectedKernel::GridSize(problem.M, problem.N, problem.K);
        dim3 blocks      = SelectedKernel::BlockSize();
        size_t lds_bytes = SelectedKernel::GetSmemSize();

        // Time kernel execution
        hipEvent_t start, stop;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);

        (void)hipEventRecord(start, hip_stream);

        // Launch kernel
        ck_tile::launch_kernel(SelectedKernel::Kernel, grids, blocks, lds_bytes, hip_stream, kargs);

        (void)hipEventRecord(stop, hip_stream);
        (void)hipEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        (void)hipEventElapsedTime(&elapsed_ms, start, stop);

        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);

        return elapsed_ms;
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        // Use validation helper
        using ADataType   = typename SelectedKernel::ADataType;
        using BDataType   = typename SelectedKernel::BDataType;
        using CDataType   = typename SelectedKernel::CDataType;
        using AccDataType = typename SelectedKernel::AccDataType;

        // d_ptrs not yet supported
        (void)d_ptrs;

        // Convert tolerance to rtol and atol
        float rtol = tolerance;
        float atol = tolerance * 1e-2f; // atol is typically smaller

        return validation::validate_gemm_kernel<ADataType, BDataType, CDataType, AccDataType>(
            a_ptr, b_ptr, c_ptr, problem, rtol, atol);
    }

    private:
    int64_t estimate_smem_usage() const
    {
        // Use kernel's reported shared memory size
        return SelectedKernel::GetSmemSize();
    }

    KernelKey key_;
    std::string name_;
};

/// Helper function to create a tile kernel instance wrapper
/// This should be called from generated code that knows the SelectedKernel type
template <typename SelectedKernel>
std::shared_ptr<KernelInstance> create_tile_kernel_instance(const KernelKey& key,
                                                            const std::string& name)
{
    return std::make_shared<TileKernelInstance<SelectedKernel>>(key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
