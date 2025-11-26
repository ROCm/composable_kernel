// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: Multi-D GEMM
 *
 * Demonstrates Multi-D GEMM which fuses additional elementwise operations
 * with the matrix multiplication, such as bias addition and activations.
 *
 * Complexity: ★★★★★
 *
 * Multi-D GEMM Overview:
 *   Standard GEMM: C = A @ B
 *   Multi-D GEMM:  C = ElementwiseOp(A @ B, D0, D1, ...)
 *
 * Supported Elementwise Operations:
 *   - PassThrough: C = A @ B (no fusion)
 *   - MultiDAdd: C = A @ B + D0 + D1 + ... (bias addition)
 *   - Relu: C = relu(A @ B + D0)
 *   - Gelu: C = gelu(A @ B + D0)
 *   - Sigmoid: C = sigmoid(A @ B + D0)
 *   - Tanh: C = tanh(A @ B + D0)
 *   - Swish: C = swish(A @ B + D0)
 *   - HardSwish: C = hardswish(A @ B + D0)
 *
 * Use Cases:
 *   - Fused linear layers with bias: Y = XW + b
 *   - Activation fusion: Y = relu(XW + b)
 *   - Residual connections: Y = XW + residual
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL CONFIGURATION - Multi-D with Bias
// =============================================================================

namespace multi_d_config {

using ADataType   = ck_tile::fp16_t;
using BDataType   = ck_tile::fp16_t;
using CDataType   = ck_tile::fp16_t;
using DDataType   = ck_tile::fp16_t; // Bias/residual type
using AccDataType = float;

constexpr int TileM = 128;
constexpr int TileN = 128;
constexpr int TileK = 32;

constexpr int WavesM = 2;
constexpr int WavesN = 2;
constexpr int WavesK = 1;

constexpr int WarpM = 32;
constexpr int WarpN = 32;
constexpr int WarpK = 16;

constexpr int BlockSize = 256;

} // namespace multi_d_config

// =============================================================================
// Helper: Configure Multi-D kernel key
// =============================================================================

KernelKey make_multi_d_key(int num_d_tensors, const std::string& elementwise_op)
{
    using namespace multi_d_config;

    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();

    // Tile configuration (same as standard)
    builder.tile_m = TileM;
    builder.tile_n = TileN;
    builder.tile_k = TileK;

    builder.wave_m = WavesM;
    builder.wave_n = WavesN;
    builder.wave_k = WavesK;

    builder.warp_m = WarpM;
    builder.warp_n = WarpN;
    builder.warp_k = WarpK;

    builder.block_size = BlockSize;

    // Multi-D specific configuration
    builder.num_d_tensors  = num_d_tensors;
    builder.elementwise_op = elementwise_op;

    return builder.build();
}

// =============================================================================
// CPU Reference for Multi-D operations
// =============================================================================

template <typename T>
void cpu_relu(T* data, int64_t size)
{
    for(int64_t i = 0; i < size; ++i)
    {
        float val = static_cast<float>(data[i]);
        data[i]   = static_cast<T>(val > 0 ? val : 0);
    }
}

template <typename T>
void cpu_gelu(T* data, int64_t size)
{
    // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float c = 0.7978845608f; // sqrt(2/π)
    constexpr float d = 0.044715f;
    for(int64_t i = 0; i < size; ++i)
    {
        float x     = static_cast<float>(data[i]);
        float inner = c * (x + d * x * x * x);
        data[i]     = static_cast<T>(0.5f * x * (1.0f + std::tanh(inner)));
    }
}

template <typename T>
void cpu_sigmoid(T* data, int64_t size)
{
    for(int64_t i = 0; i < size; ++i)
    {
        float x = static_cast<float>(data[i]);
        data[i] = static_cast<T>(1.0f / (1.0f + std::exp(-x)));
    }
}

template <typename T>
void cpu_add_bias(T* output, const T* bias, int64_t M, int64_t N)
{
    // Add bias (broadcast over M dimension)
    for(int64_t m = 0; m < M; ++m)
    {
        for(int64_t n = 0; n < N; ++n)
        {
            float val = static_cast<float>(output[m * N + n]);
            val += static_cast<float>(bias[n]);
            output[m * N + n] = static_cast<T>(val);
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv)
{
    print_header("Example 08: Multi-D GEMM");

    using namespace multi_d_config;

    // Parse problem size
    int M = 1024, N = 1024, K = 1024;
    if(argc >= 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::cout << "Problem: " << format_size(M, N, K) << "\n\n";

    // -------------------------------------------------------------------------
    // Explain Multi-D GEMM operations
    // -------------------------------------------------------------------------
    std::cout << "Multi-D GEMM Operations:\n";
    print_separator('-', 60);

    struct OpInfo
    {
        const char* name;
        const char* formula;
        int num_d;
    };

    std::vector<OpInfo> operations = {
        {"PassThrough", "C = A @ B", 0},
        {"MultiDAdd", "C = A @ B + D0 + D1 + ...", 1},
        {"Relu", "C = relu(A @ B + D0)", 1},
        {"Gelu", "C = gelu(A @ B + D0)", 1},
        {"Sigmoid", "C = sigmoid(A @ B + D0)", 1},
        {"Tanh", "C = tanh(A @ B + D0)", 1},
        {"Swish", "C = x * sigmoid(x), x=A@B+D0", 1},
    };

    for(const auto& op : operations)
    {
        std::cout << "  " << op.name << ": " << op.formula << "\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Demonstrate configuration for each operation
    // -------------------------------------------------------------------------
    std::cout << "Key Configuration Examples:\n";
    print_separator('-', 60);

    // Standard GEMM
    {
        KernelKey key = make_multi_d_key(0, "PassThrough");
        std::cout << "1. Standard GEMM (no fusion):\n";
        std::cout << "   num_d_tensors: " << key.signature.num_d_tensors << "\n";
        std::cout << "   elementwise_op: " << key.signature.elementwise_op << "\n\n";
    }

    // GEMM + Bias
    {
        KernelKey key = make_multi_d_key(1, "MultiDAdd");
        std::cout << "2. GEMM with Bias (C = A @ B + bias):\n";
        std::cout << "   num_d_tensors: " << key.signature.num_d_tensors << "\n";
        std::cout << "   elementwise_op: " << key.signature.elementwise_op << "\n\n";
    }

    // GEMM + Bias + ReLU
    {
        KernelKey key = make_multi_d_key(1, "Relu");
        std::cout << "3. GEMM with Bias and ReLU (C = relu(A @ B + bias)):\n";
        std::cout << "   num_d_tensors: " << key.signature.num_d_tensors << "\n";
        std::cout << "   elementwise_op: " << key.signature.elementwise_op << "\n\n";
    }

    // GEMM + Bias + GELU (common in transformers)
    {
        KernelKey key = make_multi_d_key(1, "Gelu");
        std::cout << "4. GEMM with Bias and GELU (Transformer FFN):\n";
        std::cout << "   num_d_tensors: " << key.signature.num_d_tensors << "\n";
        std::cout << "   elementwise_op: " << key.signature.elementwise_op << "\n\n";
    }

    // -------------------------------------------------------------------------
    // Generate kernels instructions
    // -------------------------------------------------------------------------
    print_separator('-', 60);
    std::cout << "To generate Multi-D kernels:\n\n";
    std::cout << "  cd dispatcher/codegen\n";
    std::cout << "  python3 unified_gemm_codegen.py \\\n";
    std::cout << "    --elementwise MultiDAdd \\\n";
    std::cout << "    --num-d-tensors 1 \\\n";
    std::cout << "    --output-dir ../build/generated_kernels\n\n";

    std::cout << "For activation fusion:\n";
    std::cout << "  python3 unified_gemm_codegen.py \\\n";
    std::cout << "    --elementwise Relu \\\n";
    std::cout << "    --num-d-tensors 1\n\n";
    print_separator('-', 60);

    // -------------------------------------------------------------------------
    // Fallback demonstration with standard kernel
    // -------------------------------------------------------------------------
    std::cout << "\nDemonstrating with standard kernel (no fusion)...\n\n";

    // Use standard kernel
    KernelKeyBuilder fallback = KernelKeyBuilder::fp16_rcr();
    fallback.tile_m           = SelectedKernel::TileM;
    fallback.tile_n           = SelectedKernel::TileN;
    fallback.tile_k           = SelectedKernel::TileK;
    fallback.wave_m           = SelectedKernel::WarpPerBlock_M;
    fallback.wave_n           = SelectedKernel::WarpPerBlock_N;
    fallback.wave_k           = SelectedKernel::WarpPerBlock_K;
    fallback.warp_m           = SelectedKernel::WarpTileM;
    fallback.warp_n           = SelectedKernel::WarpTileN;
    fallback.warp_k           = SelectedKernel::WarpTileK;
    fallback.block_size       = SelectedKernel::BlockSize;

    KernelKey key = fallback.build();

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, "fp16_rcr_standard");

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    // Allocate memory
    Problem problem(M, N, K);

    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K);
    std::vector<BDataType> b_host(K * N);
    std::vector<DDataType> bias(N);

    // Initialize
    fill_random(a_host.data(), M * K, ADataType(-0.5f), ADataType(0.5f));
    fill_random(b_host.data(), K * N, BDataType(-0.5f), BDataType(0.5f));
    fill_random(bias.data(), N, DDataType(-0.1f), DDataType(0.1f));

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    // Run standard GEMM
    Dispatcher dispatcher;
    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    std::cout << "Step 1: Standard GEMM (C = A @ B)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms)
              << "\n\n";

    // Simulate bias addition on CPU (what Multi-D would fuse)
    std::vector<CDataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    std::cout << "Step 2: Adding bias on CPU (simulating Multi-D fusion)\n";
    Timer cpu_timer;
    cpu_timer.start();
    cpu_add_bias(c_host.data(), bias.data(), M, N);
    double bias_time = cpu_timer.elapsed_ms();
    std::cout << "  Bias time: " << std::fixed << std::setprecision(4) << bias_time << " ms\n\n";

    std::cout << "Step 3: Applying ReLU on CPU (simulating activation fusion)\n";
    cpu_timer.start();
    cpu_relu(c_host.data(), M * N);
    double relu_time = cpu_timer.elapsed_ms();
    std::cout << "  ReLU time: " << std::fixed << std::setprecision(4) << relu_time << " ms\n\n";

    // Summary
    print_separator('-', 60);
    std::cout << "Performance Summary:\n";
    std::cout << "  Unfused (GEMM + Bias + ReLU): " << std::fixed << std::setprecision(4)
              << (time_ms + bias_time + relu_time) << " ms\n";
    std::cout << "  With Multi-D fusion: ~" << time_ms << " ms (estimated)\n";
    std::cout << "  Potential speedup: " << std::setprecision(1)
              << ((time_ms + bias_time + relu_time) / time_ms) << "x\n\n";

    print_separator();
    std::cout << "Multi-D example complete!\n";
    std::cout << "(Note: Actual Multi-D kernels require separate generation)\n";
    print_separator();

    return 0;
}
