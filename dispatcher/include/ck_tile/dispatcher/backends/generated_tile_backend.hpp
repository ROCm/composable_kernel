// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <sstream>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/**
 * Kernel instance wrapper for unified_gemm_codegen.py generated kernels
 * 
 * These kernels have structure:
 * - Types defined outside: using ADataType = ...; using BDataType = ...;
 * - struct SelectedKernel with static constexpr config and launch() method
 * - constexpr const char* KERNEL_NAME = "...";
 * 
 * This is different from tile_engine style where everything is in SelectedKernel.
 */
template <typename SelectedKernelType,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_>
class GeneratedTileKernelInstance : public KernelInstance
{
public:
    using ADataType = ADataType_;
    using BDataType = BDataType_;
    using CDataType = CDataType_;
    using AccDataType = AccDataType_;
    using SelectedKernel = SelectedKernelType;
    
    GeneratedTileKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    bool supports(const Problem& problem) const override
    {
        // Check dimension divisibility if padding not enabled
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;

        if(pad_m && pad_n && pad_k)
        {
            return true;  // Padding enabled - supports any size
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
        (void)d_ptrs;  // Not used in basic GEMM
        
        // Create arguments structure
        ck_tile::GemmHostArgs args;
        args.a_ptr = const_cast<void*>(a_ptr);
        args.b_ptr = const_cast<void*>(b_ptr);
        args.c_ptr = c_ptr;
        args.M = problem.M;
        args.N = problem.N;
        args.K = problem.K;
        args.k_batch = problem.k_batch;
        
        // Create stream config
        ck_tile::stream_config stream_cfg;
        stream_cfg.stream_id_ = reinterpret_cast<hipStream_t>(stream);
        stream_cfg.time_kernel_ = true;
        stream_cfg.log_level_ = 0;
        
        // Call the generated kernel's launch method
        return SelectedKernel::launch(args, stream_cfg);
    }

    bool validate(const void* a_ptr,
                 const void* b_ptr,
                 const void* c_ptr,
                 const void** d_ptrs,
                 const Problem& problem,
                 float tolerance) const override
    {
        (void)a_ptr; (void)b_ptr; (void)c_ptr; (void)d_ptrs;
        (void)problem; (void)tolerance;
        // Validation would require reference implementation
        return true;
    }

private:
    KernelKey key_;
    std::string name_;
};

/// Helper function to create a generated tile kernel instance wrapper
template <typename SelectedKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType>
std::shared_ptr<KernelInstance> create_generated_tile_kernel(
    const KernelKey& key,
    const std::string& name)
{
    return std::make_shared<GeneratedTileKernelInstance<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType>>(key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile

