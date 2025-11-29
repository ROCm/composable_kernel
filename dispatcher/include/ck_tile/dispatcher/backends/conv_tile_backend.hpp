// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/dispatcher/conv_problem.hpp"
#include "ck_tile/dispatcher/conv_registry.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include <hip/hip_runtime.h>
#include <numeric>
#include <iostream>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// =============================================================================
// ConvHostArgs - Host-side convolution arguments
// =============================================================================

struct ConvHostArgs
{
    // Pointers
    const void* input_ptr;
    const void* weight_ptr;
    void* output_ptr;

    // Dimensions
    ck_tile::index_t N; // Batch
    ck_tile::index_t G; // Groups
    ck_tile::index_t C; // Input channels
    ck_tile::index_t K; // Output channels

    // Spatial dimensions
    std::vector<ck_tile::index_t> input_spatial;
    std::vector<ck_tile::index_t> filter_spatial;
    std::vector<ck_tile::index_t> output_spatial;

    // Convolution parameters
    std::vector<ck_tile::index_t> strides;
    std::vector<ck_tile::index_t> paddings;
    std::vector<ck_tile::index_t> dilations;

    // Split-K
    ck_tile::index_t k_batch = 1;

    ConvHostArgs() = default;

    ConvHostArgs(const ConvProblem& prob, const void* in, const void* wei, void* out)
        : input_ptr(in),
          weight_ptr(wei),
          output_ptr(out),
          N(prob.N),
          G(prob.G),
          C(prob.C),
          K(prob.K),
          k_batch(1)
    {

        // Copy spatial dimensions
        for(int i = 0; i < 3; ++i)
        {
            if(prob.input_spatial[i] > 1 || i == 2)
            {
                input_spatial.push_back(prob.input_spatial[i]);
                filter_spatial.push_back(prob.filter_spatial[i]);
                output_spatial.push_back(prob.output_spatial[i]);
                strides.push_back(prob.stride[i]);
                paddings.push_back(prob.padding[i]);
                dilations.push_back(prob.dilation[i]);
            }
        }
    }

    ck_tile::index_t num_spatial_dims() const { return input_spatial.size(); }

    // Get effective GemmM (output spatial product * N)
    ck_tile::index_t get_gemm_m() const
    {
        ck_tile::index_t spatial_product = 1;
        for(auto s : output_spatial)
            spatial_product *= s;
        return N * spatial_product;
    }

    // Get effective GemmN (K)
    ck_tile::index_t get_gemm_n() const { return K; }

    // Get effective GemmK (C * filter spatial product)
    ck_tile::index_t get_gemm_k() const
    {
        ck_tile::index_t filter_product = 1;
        for(auto f : filter_spatial)
            filter_product *= f;
        return C * filter_product;
    }

    // FLOPs calculation
    double get_flops() const { return 2.0 * G * get_gemm_m() * get_gemm_n() * get_gemm_k(); }
};

// =============================================================================
// ConvTileKernelInstance - Kernel instance for CK Tile convolutions
// =============================================================================

template <typename ConvConfig>
class ConvTileKernelInstance : public ConvKernelInstance
{
    public:
    using InDataType  = typename ConvConfig::InDataType;
    using WeiDataType = typename ConvConfig::WeiDataType;
    using OutDataType = typename ConvConfig::OutDataType;
    using AccDataType = typename ConvConfig::AccDataType;

    ConvTileKernelInstance(const ConvKernelKey& key, const std::string& name)
        : ConvKernelInstance(key, name, [this](const ConvProblem& prob, void* stream) {
              return this->launch(prob, stream);
          })
    {
    }

    float launch(const ConvProblem& problem, void* stream) const
    {
        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

        // Allocate device memory
        size_t input_size  = problem.N * problem.G * problem.C;
        size_t weight_size = problem.G * problem.K * problem.C;
        size_t output_size = problem.N * problem.G * problem.K;

        for(int i = 0; i < 3; ++i)
        {
            if(problem.input_spatial[i] > 1)
            {
                input_size *= problem.input_spatial[i];
            }
            if(problem.filter_spatial[i] > 1)
            {
                weight_size *= problem.filter_spatial[i];
            }
            if(problem.output_spatial[i] > 1)
            {
                output_size *= problem.output_spatial[i];
            }
        }

        // For now, return placeholder timing
        // Full implementation requires proper kernel instantiation
        std::cout << "  ConvTileKernelInstance::launch()\n";
        std::cout << "    GemmM: " << problem.N * problem.Ho() * problem.Wo() << "\n";
        std::cout << "    GemmN: " << problem.K << "\n";
        std::cout << "    GemmK: " << problem.C * problem.Y() * problem.X() << "\n";

        return 0.0f;
    }
};

// =============================================================================
// Helper to create ConvKernelInstance from ConvConfig
// =============================================================================

template <typename ConvConfig>
std::shared_ptr<ConvKernelInstance> create_conv_kernel_instance(const std::string& name,
                                                                ConvOp op = ConvOp::Forward)
{

    ConvKernelKey key;
    key.dtype_in     = "fp16"; // Would extract from ConvConfig::InDataType
    key.dtype_wei    = "fp16";
    key.dtype_out    = "fp16";
    key.ndim_spatial = ConvConfig::NDimSpatial;
    key.op           = op;
    key.tile_m       = ConvConfig::M_Tile;
    key.tile_n       = ConvConfig::N_Tile;
    key.tile_k       = ConvConfig::K_Tile;
    key.pipeline     = "compv4"; // Would extract from ConvConfig::Pipeline
    key.scheduler    = "intrawave";

    return std::make_shared<ConvTileKernelInstance<ConvConfig>>(key, name);
}

// =============================================================================
// Simple Conv Runner - For quick testing without full dispatcher
// =============================================================================

template <typename InDataType, typename WeiDataType, typename OutDataType>
class SimpleConvRunner
{
    public:
    SimpleConvRunner() = default;

    float run_forward_2d(const InDataType* input,
                         const WeiDataType* weight,
                         OutDataType* output,
                         const ConvProblem& problem,
                         hipStream_t stream = nullptr)
    {

        // Create host args
        ConvHostArgs args(problem, input, weight, output);

        std::cout << "SimpleConvRunner::run_forward_2d()\n";
        std::cout << "  Input:  N=" << problem.N << " C=" << problem.C << " H=" << problem.Hi()
                  << " W=" << problem.Wi() << "\n";
        std::cout << "  Weight: K=" << problem.K << " C=" << problem.C << " Y=" << problem.Y()
                  << " X=" << problem.X() << "\n";
        std::cout << "  Output: N=" << problem.N << " K=" << problem.K << " Ho=" << problem.Ho()
                  << " Wo=" << problem.Wo() << "\n";
        std::cout << "  FLOPs:  " << std::scientific << args.get_flops() << "\n";

        // For now, return placeholder - full implementation would use CK Tile kernel
        return 0.0f;
    }
};

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
