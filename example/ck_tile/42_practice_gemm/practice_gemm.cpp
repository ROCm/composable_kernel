// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <iostream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "practice_gemm.hpp"
#include "reference_gemm.hpp"

int main()
{
    // TODO: GemmTypeConfig
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = float;
    using AccDataType = float;

    // ArgParser
    ck_tile::index_t M            = 1024;
    ck_tile::index_t N            = 1024;
    ck_tile::index_t K            = 1024;
    ck_tile::index_t verification = 1;

    ck_tile::index_t stride_a = K;
    ck_tile::index_t stride_b = K;
    ck_tile::index_t stride_c = N;

    auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    // B is treated as [N, K] in reference and device views
    auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};

    auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};
    auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};
    auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};

    // tensors on host (cpu)
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host(c_lengths, c_strides);

    // initialize tensors
    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // Print the tensors using the new print_first_n member function
    // std::cout << "Tensor A (first 5 elements): ";
    // a_host.print_first_n(5);
    // std::cout << std::endl;

    // std::cout << "Tensor B (first 5 elements): ";
    // b_host.print_first_n(5);
    // std::cout << std::endl;

    // std::cout << "Tensor C (first 5 elements): ";
    // c_host.print_first_n(5);
    // std::cout << std::endl;

    // Create device tensors of same size as host tensors and copy data
    ck_tile::DeviceMem a_device(a_host);
    ck_tile::DeviceMem b_device(b_host);
    ck_tile::DeviceMem c_device(c_host);

    (void)verification;
    (void)AccDataType{}; // Fake usage to suppress unused warning

    // TODO: BlockTileConfig
    // constexpr ck_tile::index_t warpSize    = 64;
    constexpr ck_tile::index_t kBlockSize = 256;

    using BlockTile = ck_tile::sequence<256, 32, 64>;
    using WaveTile  = ck_tile::sequence<16, 16, 16>;

    std::cout << "Creating PracticeGemmShape, PracticeGemmProblem, PracticeGemmPolicy" << std::endl;
    using PracticeGemmShape       = ck_tile::PracticeGemmShape<BlockTile, WaveTile>;
    using PracticeGemmHostProblem = ck_tile::
        PracticeGemmHostProblem<ADataType, BDataType, CDataType, AccDataType, PracticeGemmShape>;
    using PracticeGemmHostPolicy = ck_tile::PracticeGemmHostPolicy;

    ck_tile::index_t kGridSize = ck_tile::integer_divide_ceil(M, PracticeGemmShape::BlockTile_M) *
                                 ck_tile::integer_divide_ceil(N, PracticeGemmShape::BlockTile_N);

    std::cout << "kGridSize: " << kGridSize << std::endl;

    constexpr ck_tile::index_t kWarpPerCU    = 8; // two warps per CU
    constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / ck_tile::get_warp_size();
    constexpr ck_tile::index_t kBlockPerCU   = kWarpPerCU / kWarpPerBlock;

    std::cout << "kBlockSize: " << kBlockSize << std::endl;
    std::cout << "kWarpPerBlock: " << kWarpPerBlock << std::endl;
    std::cout << "kBlockPerCU: " << kBlockPerCU << std::endl;

    std::cout << "PracticeGemmShape: " << PracticeGemmShape::GetName() << std::endl;

    using gemm_kernel =
        ck_tile::PracticeGemmKernel<PracticeGemmHostProblem, PracticeGemmHostPolicy>;
    static_cast<void>(sizeof(gemm_kernel));

    float ave_time = ck_tile::launch_kernel(ck_tile::stream_config{nullptr, true, 0, 5, 1000},
                                            ck_tile::make_kernel<kBlockSize, kBlockPerCU>(
                                                gemm_kernel{},
                                                kGridSize,
                                                kBlockSize,
                                                0,
                                                static_cast<ADataType*>(a_device.GetDeviceBuffer()),
                                                static_cast<BDataType*>(b_device.GetDeviceBuffer()),
                                                static_cast<CDataType*>(c_device.GetDeviceBuffer()),
                                                M,
                                                N,
                                                K,
                                                stride_a,
                                                stride_b,
                                                stride_c));

    auto pass = true;

    if(verification)
    {
        // reference gemm
        ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_basic_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_host, c_host_ref);
        ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);
        c_device.FromDevice(c_host_dev.data());
        pass &= ck_tile::check_err(c_host_dev, c_host_ref);
        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !pass;
}
