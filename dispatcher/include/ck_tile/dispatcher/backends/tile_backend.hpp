// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/dispatcher/backends/backend_base.hpp"
#include "ck_tile/dispatcher/validation/reference_kernels.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// Kernel instance for CK Tile generated kernels
template <typename SelectedKernel>
class TileKernelInstance : public KernelInstance
{
public:
    TileKernelInstance(const KernelKey& key, const std::string& name)
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
             const Problem& problem,
             hipStream_t stream = nullptr) override
    {
        // Construct kernel arguments
        using ADataType = typename SelectedKernel::ADataType;
        using BDataType = typename SelectedKernel::BDataType;
        using CDataType = typename SelectedKernel::CDataType;

        auto kargs = SelectedKernel::MakeKernelArgs(
            static_cast<const ADataType*>(a_ptr),
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
        dim3 grids     = SelectedKernel::GridSize(problem.M, problem.N, problem.K);
        dim3 blocks    = SelectedKernel::BlockSize();
        size_t lds_bytes = SelectedKernel::GetSmemSize();

        // Time kernel execution
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, stream);

        // Launch kernel
        ck_tile::launch_kernel(
            SelectedKernel::Kernel, grids, blocks, lds_bytes, stream, kargs);

        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        hipEventElapsedTime(&elapsed_ms, start, stop);

        hipEventDestroy(start);
        hipEventDestroy(stop);

        return elapsed_ms;
    }

    BackendType get_backend_type() const override { return BackendType::Tile; }

    std::string get_metadata() const override
    {
        std::ostringstream oss;
        oss << KernelInstance::get_metadata()
            << ",tile=" << SelectedKernel::TileM << "x" << SelectedKernel::TileN << "x"
            << SelectedKernel::TileK
            << ",block_size=" << SelectedKernel::BlockSize
            << ",persistent=" << (SelectedKernel::UsePersistentKernel ? "true" : "false");
        return oss.str();
    }

    bool validate(const void* a_ptr,
                 const void* b_ptr,
                 const void* c_ptr,
                 const Problem& problem,
                 float rtol = 1e-3f,
                 float atol = 1e-5f) const override
    {
        // Use validation helper
        using ADataType = typename SelectedKernel::ADataType;
        using BDataType = typename SelectedKernel::BDataType;
        using CDataType = typename SelectedKernel::CDataType;
        using AccDataType = typename SelectedKernel::AccDataType;
        
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

/// Backend for CK Tile generated kernels
class TileBackend : public BackendBase
{
public:
    TileBackend() = default;

    std::vector<std::shared_ptr<KernelInstance>>
    discover_kernels(const std::string& search_path) override
    {
        std::vector<std::shared_ptr<KernelInstance>> kernels;

        namespace fs = std::filesystem;

        if(!fs::exists(search_path))
        {
            return kernels;
        }

        // Scan for generated header files
        for(const auto& entry : fs::recursive_directory_iterator(search_path))
        {
            if(entry.is_regular_file() && entry.path().extension() == ".hpp")
            {
                try
                {
                    auto kernel = parse_kernel_header(entry.path().string());
                    if(kernel)
                    {
                        kernels.push_back(kernel);
                    }
                }
                catch(const std::exception& e)
                {
                    // Skip files that can't be parsed
                    continue;
                }
            }
        }

        return kernels;
    }

    std::shared_ptr<KernelInstance>
    create_kernel_instance(const KernelKey& kernel_key) override
    {
        // This would create a kernel instance from a KernelKey
        // For now, throw as this requires template instantiation
        throw std::runtime_error(
            "create_kernel_instance not yet implemented for TileBackend");
    }

    BackendType get_backend_type() const override { return BackendType::Tile; }

private:
    std::shared_ptr<KernelInstance> parse_kernel_header(const std::string& header_path)
    {
        std::ifstream file(header_path);
        if(!file.is_open())
        {
            return nullptr;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        // Extract kernel name
        std::regex kernel_name_regex(R"(constexpr const char\* KERNEL_NAME\s*=\s*"([^"]+)")");
        std::smatch match;

        if(!std::regex_search(content, match, kernel_name_regex))
        {
            return nullptr;
        }

        std::string kernel_name = match[1].str();

        // Extract tile configuration
        int tile_m = extract_constexpr_int(content, "TileM");
        int tile_n = extract_constexpr_int(content, "TileN");
        int tile_k = extract_constexpr_int(content, "TileK");

        if(tile_m == 0 || tile_n == 0 || tile_k == 0)
        {
            return nullptr;
        }

        // Build KernelKey (simplified - would need full parsing)
        KernelKey key;
        key.signature.dtype_a    = DataType::FP16;
        key.signature.dtype_b    = DataType::FP16;
        key.signature.dtype_c    = DataType::FP16;
        key.signature.dtype_acc  = DataType::FP32;
        key.signature.layout_a   = LayoutTag::RowMajor;
        key.signature.layout_b   = LayoutTag::ColMajor;
        key.signature.layout_c   = LayoutTag::RowMajor;
        key.algorithm.tile_shape = {static_cast<uint16_t>(tile_m),
                                   static_cast<uint16_t>(tile_n),
                                   static_cast<uint16_t>(tile_k)};
        key.gfx_arch             = 942;

        // Note: This returns nullptr because we can't instantiate the template
        // without knowing the SelectedKernel type at compile time.
        // In practice, kernels would be registered explicitly in generated code.
        return nullptr;
    }

    int extract_constexpr_int(const std::string& content, const std::string& name)
    {
        std::string pattern = R"(constexpr\s+(?:static\s+)?(?:const\s+)?(?:int|std::size_t|auto)\s+)" +
                             name + R"(\s*=\s*(\d+))";
        std::regex regex(pattern);
        std::smatch match;

        if(std::regex_search(content, match, regex))
        {
            return std::stoi(match[1].str());
        }

        return 0;
    }
};

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
