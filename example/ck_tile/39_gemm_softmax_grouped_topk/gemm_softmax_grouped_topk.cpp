// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstring>

#include "config.h"
#include "ck_tile/host.hpp"
#include "gemm_softmax_grouped_topk.hpp"
#include "reference_gemm_softmax_grouped_topk.hpp"

/*
 * Toy code of GEMM
 * Assume simplest case.
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

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

int main(int argc, char* argv[])
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    using WeightType  = float;
    using IndexType   = ck_tile::index_t;

    ck_tile::index_t verification = 0;
    ck_tile::index_t M            = 3328;
    ck_tile::index_t N            = 4096;
    ck_tile::index_t K            = 4096;
    ck_tile::index_t topk         = 16;

    if(argc == 2)
    {
        verification = std::stoi(argv[1]);
    }
    if(argc == 6)
    {
        verification = std::stoi(argv[1]);
        M            = std::stoi(argv[2]);
        N            = std::stoi(argv[3]);
        K            = std::stoi(argv[4]);
        topk         = std::stoi(argv[5]);
    }

#if defined(KERNEL_A)
    printf("*** Kernel A test ***  \n");
    printf("  --> Using mfma_32x32x(8x2)\n");
#elif defined(KERNEL_B)
    printf("*** Kernel B test ***  \n");
    printf("  --> Using mfma_16x16x16\n");
#elif defined(KERNEL_C)
    printf("*** Kernel C test ***  \n");
    printf("  --> Using mfma_16x16x(16x2)\n");
#elif defined(KERNEL_D)
    printf("*** Kernel D test ***  \n");
    printf("  --> Using mfma_16x16x(16x2)\n");
    printf("  --> XOR-based bank-conflict-free\n");
#elif defined(KERNEL_E)
    printf("*** Kernel E test ***\n");
    printf("  --> Using mfma_16x16x(16x2)\n");
    printf("  --> XOR-based bank-conflict-free\n");
    printf("  --> Adjust block tile shape\n");
#elif defined(KERNEL_F)
    printf("*** Kernel F test ***\n");
    printf("  --> Using mfma_16x16x(16x2)\n");
    printf("  --> XOR-based bank-conflict-free\n");
    printf("  --> Adjust block tile shape\n");
    printf("  --> Enable prefetch\n");
#elif defined(KERNEL_G)
    printf("*** Kernel G test ***\n");
    printf("  --> Using mfma_16x16x(16x2)\n");
    printf("  --> XOR-based bank-conflict-free\n");
    printf("  --> Adjust block tile shape\n");
    printf("  --> Enable prefetch\n");
    printf("  --> Enable instruction schedule\n");
#elif defined(KERNEL_H)
    printf("*** Kernel H test ***\n");
    printf("  --> Using mfma_16x16x(16x2)\n");
    printf("  --> XOR-based bank-conflict-free\n");
    printf("  --> Adjust block tile shape\n");
    printf("  --> Enable prefetch\n");
    printf("  --> Enable instruction schedule\n");
    printf("  --> Enable cache-aware thread blocks schedule\n");
#else
    printf("*** Naive implementation test ***\n");
#endif

    const ck_tile::index_t Lda = K;
    const ck_tile::index_t Ldb = K;
    // const ck_tile::index_t Ldc = N;
    const ck_tile::index_t Ldout = topk;

    const auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    const auto a_strides = std::array<ck_tile::index_t, 2>{Lda, 1};

    const auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    const auto b_strides = std::array<ck_tile::index_t, 2>{Ldb, 1};

    // const auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};
    // const auto c_strides = std::array<ck_tile::index_t, 2>{Ldc, 1};

    const auto out_lengths = std::array<ck_tile::index_t, 2>{M, topk};
    const auto out_strides = std::array<ck_tile::index_t, 2>{Ldout, 1};

    // host verify
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    // ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);
    ck_tile::HostTensor<WeightType> value_host_dev(out_lengths, out_strides);
    ck_tile::HostTensor<IndexType> index_host_dev(out_lengths, out_strides);

    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    // ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_buf(value_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_buf(index_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    // Alignment
    constexpr ck_tile::index_t kAAlignment = 8;
    constexpr ck_tile::index_t kBAlignment = 8;
    // constexpr ck_tile::index_t kCAlignment = 8;
    constexpr ck_tile::index_t kOutAlignment = 8;

    constexpr ck_tile::index_t kBlockSize = 256;

#ifdef ADJUST_BLOCK_TILE_SHAPE
    constexpr ck_tile::index_t kGemmMPerBlock = 128;
    constexpr ck_tile::index_t kGemmKPerBlock = 64;
#else
    constexpr ck_tile::index_t kGemmMPerBlock = 128;
    constexpr ck_tile::index_t kGemmKPerBlock = 16;
#endif
    constexpr ck_tile::index_t kGemmNPerBlock = 256;
    constexpr ck_tile::index_t kGemmTopKPerBlock = 16;

    ck_tile::index_t kGridSize = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck_tile::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    using gemm_kernel = ck_tile::Gemm<ADataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      WeightType,
                                      IndexType,
                                      CElementFunction,
                                      kAAlignment,
                                      kBAlignment,
                                      kOutAlignment,
                                      kBlockSize,
                                      kGemmMPerBlock,
                                      kGemmNPerBlock,
                                      kGemmKPerBlock,
                                      kGemmTopKPerBlock>;

    float ave_time = ck_tile::launch_kernel(ck_tile::stream_config{nullptr, true, 0, 5, 1000},
                                            ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                                gemm_kernel{},
                                                kGridSize,
                                                kBlockSize,
                                                0,
                                                static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                                static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                                static_cast<WeightType*>(value_buf.GetDeviceBuffer()),
                                                static_cast<IndexType*>(index_buf.GetDeviceBuffer()),
                                                M,
                                                N,
                                                K,
                                                topk,
                                                Lda,
                                                Ldb,
                                                Ldout,
                                                CElementFunction{}));

    bool rtn = true;
    if(verification)
    {
        ck_tile::HostTensor<WeightType> value_ref(out_lengths, out_strides);
        ck_tile::HostTensor<IndexType> index_ref(out_lengths, out_strides);
        reference_basic_gemm_softmax_grouped_topk<ADataType, ADataType, AccDataType, WeightType, IndexType>(
            a_host, b_host, value_ref, index_ref, topk);
        value_buf.FromDevice(value_host_dev.mData.data());
        index_buf.FromDevice(index_host_dev.mData.data());

        // pass &= ck_tile::check_err(c_host_dev, c_host_ref);
        const ck_tile::index_t tokens = M;
        auto [rtol, atol] = get_elimit<ADataType>("");
        for(int i_t = 0; i_t < tokens; i_t++)
        {
            auto s_begin = std::vector<size_t>{static_cast<size_t>(i_t), static_cast<size_t>(0)};
            auto s_end =
                std::vector<size_t>{static_cast<size_t>(i_t + 1), static_cast<size_t>(topk)};
            auto s_value_host = value_host_dev.slice(s_begin, s_end);
            auto s_value_ref  = value_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_value_host,
                                      s_value_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Value Error:"),
                                      rtol,
                                      atol);
            auto s_index_host = index_host_dev.slice(s_begin, s_end);
            auto s_index_ref  = index_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_index_host,
                                      s_index_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Index Error:"),
                                      rtol,
                                      atol);
        }
        std::cout << "valid:" << (rtn ? "y" : "n") << std::endl;
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return rtn;
}
