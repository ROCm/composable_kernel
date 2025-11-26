// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 04: Validation
 *
 * Validates GPU GEMM results against CPU reference.
 * Note: GPU uses RCR layout (A row-major, B column-major, C row-major)
 *
 * Complexity: ★★★☆☆
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

// Reference GEMM for RCR layout (B is column-major = transposed)
template <typename AType, typename BType, typename CType>
void compute_reference_gemm_rcr(
    const AType* A, const BType* B_col_major, CType* C, int64_t M, int64_t N, int64_t K)
{
    // A is row-major: A[m,k] = A[m * K + k]
    // B is column-major: B[k,n] = B[k + n * K]  (stored transposed)
    // C is row-major: C[m,n] = C[m * N + n]
    for(int64_t m = 0; m < M; ++m)
    {
        for(int64_t n = 0; n < N; ++n)
        {
            double acc = 0;
            for(int64_t k = 0; k < K; ++k)
            {
                // B column-major: B[k,n] = B_col_major[k + n * K]
                acc +=
                    static_cast<double>(A[m * K + k]) * static_cast<double>(B_col_major[k + n * K]);
            }
            C[m * N + n] = static_cast<CType>(acc);
        }
    }
}

int main(int argc, char** argv)
{
    print_header("Example 04: Validation");

    int M = argc > 1 ? std::stoi(argv[1]) : 256;
    int N = argc > 2 ? std::stoi(argv[2]) : 256;
    int K = argc > 3 ? std::stoi(argv[3]) : 256;

    std::cout << "Problem: " << format_size(M, N, K) << "\n";
    std::cout << "Layout: RCR (A row-major, B column-major, C row-major)\n\n";

    // Setup kernel
    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();
    builder.tile_m           = SelectedKernel::TileM;
    builder.tile_n           = SelectedKernel::TileN;
    builder.tile_k           = SelectedKernel::TileK;
    builder.wave_m           = SelectedKernel::WarpPerBlock_M;
    builder.wave_n           = SelectedKernel::WarpPerBlock_N;
    builder.wave_k           = SelectedKernel::WarpPerBlock_K;
    builder.warp_m           = SelectedKernel::WarpTileM;
    builder.warp_n           = SelectedKernel::WarpTileN;
    builder.warp_k           = SelectedKernel::WarpTileK;
    builder.block_size       = SelectedKernel::BlockSize;

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            builder.build(), KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    // Allocate and initialize
    std::vector<ADataType> a_host(M * K);      // Row-major
    std::vector<BDataType> b_col_major(K * N); // Column-major (transposed)
    std::vector<CDataType> c_gpu(M * N);
    std::vector<CDataType> c_ref(M * N);

    // Fill with small random values
    fill_random(a_host.data(), M * K, ADataType(-0.1f), ADataType(0.1f));
    fill_random(b_col_major.data(), K * N, BDataType(-0.1f), BDataType(0.1f));

    // GPU execution
    std::cout << "Running GPU kernel...\n";
    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_col_major.data());
    c_dev.zero();

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    c_dev.copy_to_host(c_gpu.data());

    double tflops = calculate_tflops(M, N, K, time_ms);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms";
    std::cout << " (" << std::setprecision(2) << tflops << " TFLOPS)\n\n";

    // CPU reference with RCR layout
    std::cout << "Computing CPU reference (RCR layout)...\n";
    compute_reference_gemm_rcr(a_host.data(), b_col_major.data(), c_ref.data(), M, N, K);

    // Validate with relaxed tolerance for FP16
    std::cout << "Validating...\n";
    // rtol=0.01 (1%), atol=0.1 - relaxed for FP16
    auto result = validate_result(c_gpu.data(), c_ref.data(), M * N, 0.01, 0.1);
    result.print();

    print_separator();
    std::cout << (result.correct ? "[PASS]" : "[FAIL]") << " Validation complete!\n";
    print_separator();

    return result.correct ? 0 : 1;
}
