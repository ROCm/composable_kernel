// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>
#include "ck_tile/host.hpp"
#include "practice_gemm.hpp"
#include "../reference_gemm.hpp"

/*
 * Naive GEMM implementation (no optimizations)
 * A [M, K]
 * B [N, K]
 * C [M, N]
 */

// elementwise lambda
struct CElementFunction
{
    template <typename X>
    CK_TILE_HOST_DEVICE auto operator()(const X& x) const
    {
        return x;
    }
};

int main(int argc, char* argv[])
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;

    ck_tile::index_t verification = 0;
    ck_tile::index_t M            = 3328;
    ck_tile::index_t N            = 4096;
    ck_tile::index_t K            = 4096;

    if(argc == 2)
    {
        verification = std::stoi(argv[1]);
    }
    if(argc == 5)
    {
        verification = std::stoi(argv[1]);
        M            = std::stoi(argv[2]);
        N            = std::stoi(argv[3]);
        K            = std::stoi(argv[4]);
    }

    printf("*** Naive implementation test ***\n");

    const ck_tile::index_t Lda = K;
    const ck_tile::index_t Ldb = K;
    const ck_tile::index_t Ldc = N;

    const auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    const auto a_strides = std::array<ck_tile::index_t, 2>{Lda, 1};

    const auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    const auto b_strides = std::array<ck_tile::index_t, 2>{Ldb, 1};

    const auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};
    const auto c_strides = std::array<ck_tile::index_t, 2>{Ldc, 1};

    // host verify
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);

    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    // Alignment
    constexpr ck_tile::index_t kAAlignment = 8;
    constexpr ck_tile::index_t kBAlignment = 8;
    constexpr ck_tile::index_t kCAlignment = 8;

    constexpr ck_tile::index_t kBlockSize = 256;

    constexpr ck_tile::index_t kGemmMPerBlock = 256;
    constexpr ck_tile::index_t kGemmKPerBlock = 32;
    constexpr ck_tile::index_t kGemmNPerBlock = 128;

    ck_tile::index_t kGridSize = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck_tile::index_t kWarpSize     = 64; // AMD GPU warp size
    constexpr ck_tile::index_t kWarpPerCu    = 8;  // 2 warps per SIMD
    constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / kWarpSize;
    constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    using gemm_kernel = ck_tile::Gemm<ADataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      CElementFunction,
                                      kAAlignment,
                                      kBAlignment,
                                      kCAlignment,
                                      kBlockSize,
                                      kGemmMPerBlock,
                                      kGemmNPerBlock,
                                      kGemmKPerBlock>;

    float ave_time = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, 5, 1000},
        ck_tile::make_kernel<kBlockPerCu>(gemm_kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                          static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                          static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          Lda,
                                          Ldb,
                                          Ldc,
                                          CElementFunction{}));
    auto pass = true;

    if(verification)
    {
        // reference gemm
        ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_basic_gemm<ADataType, ADataType, AccDataType, CDataType>(
            a_host, b_host, c_host_ref);
        c_buf.FromDevice(c_host_dev.mData.data());
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
